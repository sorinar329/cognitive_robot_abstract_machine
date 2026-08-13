"""
The cramera HTTP server: static frontend + JSON API.

Serves three things from one port (default 8711):

  * the packaged web frontend (``cramera/web`` — panels, vendored libs)
  * scene bundles from :func:`cramera.paths.scenes_directory` under ``/scenes/``
  * the JSON API the panels talk to:

      GET  /api/knowledge              the knowledge-graph overview payload
      GET  /api/knowledge/view?name=   one graph tab (knowledge/kinematics/plan/chart)
      GET  /api/knowledge/expand?node= drill-down subgraph for one node
      POST /api/eql             run an EQL query string
      POST /api/model/posterior evidence-conditioned posterior for one or more
                                 variables of a causal-diagnosis model (Posterior tab)

The knowledge-graph API needs krrood (EQL); the model-query API needs
probabilistic_model. Without either, the server still serves the viewer and answers
that API's calls with ``{"ok": false, "error": ...}`` so the frontend can say why.

    cramera            # console script
    python -m cramera.server [port]
"""

from __future__ import annotations

import http.server
import json
import logging
import mimetypes
import os
import socketserver
import sys
import threading
import traceback
from pathlib import Path
from typing_extensions import Any, Callable, ClassVar, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from cramera import paths
from cramera.logging_setup import get_logger
from cramera.payload import CrameraPayload

logger = get_logger(__name__)

DEFAULT_PORT = 8711

try:
    import krrood  # noqa: F401  (the EQL engine)

    from cramera.knowledge.eql_session import EqlSession
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
    from cramera.knowledge.views.dispatcher import GraphPanelViews

    EQL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    EQL_AVAILABLE = False
    logger.warning("krrood not importable — serving the viewer without the EQL API")
except (
    Exception
):  # pragma: no cover - a broken knowledge base should not kill the viewer
    EQL_AVAILABLE = False
    traceback.print_exc()

try:
    from cramera.live.model_query import (
        EvidenceConstraint,
        EvidenceHasZeroProbability,
        ModelQueryService,
        UnknownModelVariable,
    )

    MODEL_QUERY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    MODEL_QUERY_AVAILABLE = False
    logger.warning(
        "probabilistic_model not importable — serving the viewer without the "
        "model-query API"
    )


_EQL_LOCK = threading.Lock()
"""
Krrood's SymbolGraph singleton is not threadsafe; queries are serialized.
"""

_MODEL_QUERY_SERVICE = (
    ModelQueryService(models_directory=paths.models_directory())
    if MODEL_QUERY_AVAILABLE
    else None
)
"""
Shared across requests so a model, once loaded, is not re-parsed on every query.
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    """
    Static files from the packaged web root, plus the JSON API routes.
    """

    NO_EQL_MESSAGE: ClassVar[str] = "krrood/EQL not available in this environment"
    """
    What every API route answers with when krrood is not importable.
    """

    NO_MODEL_QUERY_MESSAGE: ClassVar[str] = (
        "probabilistic_model not available in this environment"
    )
    """
    What the model-query route answers with when probabilistic_model is not importable.
    """

    def __init__(self, *args, **kwargs):
        """
        Delegate to the base handler, serving from the packaged web root.

        :param args: Positional arguments forwarded to the base handler.
        :param kwargs: Keyword arguments forwarded to the base handler.
        """
        super().__init__(*args, directory=str(paths.WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        """
        Disable caching so a rebuilt scene/frontend is never served stale.
        """
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        """
        Route the per-request access log through logging.

        :param format:``printf``-style log message format.
        :param args: Values to interpolate into ``format``.
        """
        logger.info("  " + format, *args)

    # %% helpers
    def _send_json(self, payload: Any, code: int = 200) -> None:
        """
        Send a payload as JSON with the given status code.

        ``payload`` may be a plain JSON-able dict or a :class:`CrameraPayload` — either
        is serialized the same way.

        :param payload: The payload to serialize and send.
        :param code: HTTP status code to respond with.
        """
        if isinstance(payload, CrameraPayload):
            payload = payload.to_payload()
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query_parameters(self) -> Dict[str, List[str]]:
        """
        The parsed query-string parameters of the current request.
        """
        return parse_qs(urlparse(self.path).query)

    def _guarded(self, handler: Callable[[], Any]) -> None:
        """
        Run an API handler; report exceptions as a JSON error payload.

        :param handler: The handler to run, returning the payload to send on success.
        """
        if not EQL_AVAILABLE:
            return self._send_error(self.NO_EQL_MESSAGE)
        try:
            return self._send_json(handler())
        except Exception as error:
            logger.exception("API handler failed for %s", self.path)
            return self._send_exception(error)

    # %% scene bundles (generated data, lives outside the package)
    def _serve_scene_file(self, url_path: str) -> None:
        """
        Serve one file of a scene bundle, with path-traversal protection.

        :param url_path: The request path, starting with ``/scenes/``.
        """
        relative_path = url_path[len("/scenes/") :]
        base = paths.scenes_directory().resolve()
        target = (base / relative_path).resolve()
        if not str(target).startswith(str(base) + os.sep) and target != base:
            self.send_response(403)
            self.end_headers()
            return
        if not target.is_file():
            self.send_response(404)
            self.end_headers()
            return
        content_type = (
            mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        )
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # %% routes
    def do_GET(self) -> None:
        """
        Route static files, scene bundles and the read-only API.
        """
        route = self.path.split("?")[0]
        if route.startswith("/scenes/"):
            return self._serve_scene_file(route)
        if route == "/api/knowledge":
            return self._guarded(
                lambda: GraphPanelViews.of_active_scene().for_tab("knowledge")
            )
        if route == "/api/knowledge/view":
            name = (self._query_parameters().get("name") or ["knowledge"])[0]
            return self._guarded(
                lambda: GraphPanelViews.of_active_scene().for_tab(name)
            )
        if route == "/api/knowledge/expand":
            node = (self._query_parameters().get("node") or [""])[0]
            return self._guarded(lambda: self._expanded_node(node))
        return super().do_GET()

    @staticmethod
    def _expanded_node(node: str) -> Any:
        """
        The node's subgraph, or a "not drillable" error if it has none.

        :param node: Id of the double-clicked node to expand.
        """
        payload = GraphPanelViews.of_active_scene().for_node(node)
        return payload if payload else {"ok": False, "error": "not drillable"}

    def do_POST(self) -> None:
        """
        Route the write-ish endpoints: an EQL query, or a model-posterior query.
        """
        route = self.path.split("?")[0]
        if route == "/api/eql":
            return self._run_eql_query()
        if route == "/api/model/posterior":
            return self._run_model_posterior_query()
        return self._send_error("unknown endpoint", 404)

    def _request_body(self) -> Dict[str, Any]:
        """
        The current request's JSON body, or ``{}`` if it is empty.
        """
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _run_eql_query(self) -> None:
        """
        Execute an EQL query against the active scene's knowledge base.
        """
        if not EQL_AVAILABLE:
            return self._send_error(self.NO_EQL_MESSAGE)
        try:
            code = (self._request_body().get("code") or "").strip()
            if not code:
                return self._send_error("empty query")
            with _EQL_LOCK:
                return self._send_json(EqlSession.of_active_scene().run(code))
        except Exception as error:
            # a SyntaxError from the query is named by its own type, like any other
            return self._send_exception(error)

    def _run_model_posterior_query(self) -> None:
        """
        Answer a Posterior-tab query: distribution data for one or more query variables,
        conditioned on the request's evidence.
        """
        if not MODEL_QUERY_AVAILABLE:
            return self._send_error(self.NO_MODEL_QUERY_MESSAGE)
        try:
            body = self._request_body()
            model_name = str(body.get("model") or "")
            query_variables = list(body.get("queryVariables") or [])
            evidence = [
                EvidenceConstraint.from_payload(entry)
                for entry in body.get("evidence") or []
            ]
            if not model_name or not query_variables:
                return self._send_error("model and queryVariables are required")
            payload = _MODEL_QUERY_SERVICE.posterior(
                model_name, evidence, query_variables
            )
            return self._send_json({"variables": payload})
        except (UnknownModelVariable, EvidenceHasZeroProbability) as error:
            return self._send_exception(error)
        except FileNotFoundError as error:
            return self._send_exception(error)

    def _send_error(self, message: str, code: int = 200) -> None:
        """
        Send an error payload; the panels render ``error`` as their message.

        :param message: What went wrong, in the panel's own words.
        :param code: HTTP status code to respond with.
        """
        self._send_json({"ok": False, "error": message}, code)

    def _send_exception(self, error: Exception) -> None:
        """
        Send an exception as an error payload, named by its type.

        :param error: The exception to report.
        """
        self._send_error("%s: %s" % (type(error).__name__, error))


def make_server(port: int = 0) -> socketserver.ThreadingTCPServer:
    """
    A ready-to-serve ThreadingTCPServer (port 0 = ephemeral, for tests).

    :param port: Port to listen on, or 0 for an ephemeral port.
    """
    socketserver.TCPServer.allow_reuse_address = True
    return socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)


def start_in_background(
    port: int = DEFAULT_PORT,
) -> Optional[socketserver.ThreadingTCPServer]:
    """
    Start the viewer, scenes and JSON API server on a daemon thread.

    Meant to be called from a demo file alongside
    :func:`cramera.live.runner.start`, so a single ``python3 demo.py`` brings up
    both the frontend and the live bridge. Safe to call from more than one demo
    process: if the port is already bound -- by an earlier call in this process,
    by ``cramera`` running in a separate terminal, or by another demo process --
    this logs that and returns ``None`` instead of failing the caller.

    :param port: Port to listen on.
    :return: The running server, or ``None`` if the port was already taken.
    """
    try:
        server = make_server(port)
    except OSError:
        logger.info("cramera frontend already running at http://localhost:%d/", port)
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info(
        "cramera frontend running at http://localhost:%d/", server.server_address[1]
    )
    return server


def main(arguments: Optional[List[str]] = None) -> None:
    """
    ``cramera`` — serve the viewer, the scenes and the JSON API.

    :param arguments: Command-line arguments, or None to use ``sys.argv``.
    """
    # force: an imported CRAM package may already have configured the root logger,
    # which would otherwise make this call a no-op and swallow the startup output
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    arguments = sys.argv[1:] if arguments is None else arguments
    port = int(arguments[0]) if arguments else DEFAULT_PORT
    if EQL_AVAILABLE:  # build the knowledge base once, before the first query
        EpisodeKnowledgeBase.of_active_scene()
    with make_server(port) as server:
        eql = "EQL ready (krrood)" if EQL_AVAILABLE else "EQL unavailable — static only"
        scenes = paths.scenes_directory()
        logger.info("cramera running at http://localhost:%d/ (%s)", port, eql)
        logger.info(
            "scene bundles: %s%s",
            scenes,
            "" if Path(scenes).is_dir() else "  (missing — run cramera-onboard)",
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

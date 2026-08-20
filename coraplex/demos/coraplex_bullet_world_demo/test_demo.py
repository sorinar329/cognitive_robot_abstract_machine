#!/usr/bin/env python
import traceback

try:
    import demo

    demo.main()
except Exception:
    traceback.print_exc()
    exit(1)

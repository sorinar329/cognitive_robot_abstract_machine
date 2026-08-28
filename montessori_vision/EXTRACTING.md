# This is a standalone repository, staged here

`montessori_vision` is not a member of this workspace. It is its own repository, staged on this
branch because the session that wrote it could not create a repository on GitHub — its credential
only reaches `cognitive_robot_abstract_machine` — and the container it was written in is discarded
when the session ends.

Nothing here is imported by any workspace package, and this workspace's CI runs pytest against
explicit `test/<lib>_test` paths, so this directory is not collected by it.

## Moving it to its own repository

`montessori_vision.bundle` holds the full history, so the quickest route keeps the original commit:

```bash
git clone montessori_vision.bundle montessori_vision
cd montessori_vision
git remote set-url origin git@github.com:sorinar329/montessori_vision.git   # after creating it
git push -u origin main
```

Without the bundle, the file tree beside it is the same content and can be committed fresh:

```bash
cp -r montessori_vision /somewhere/montessori_vision && cd /somewhere/montessori_vision
rm EXTRACTING.md montessori_vision.bundle
git init -b main && git add -A && git commit
```

Once it lives on GitHub, delete this directory from the branch.

## Checking it first

```bash
cd montessori_vision
pip install -e ".[dev]"
pytest
```

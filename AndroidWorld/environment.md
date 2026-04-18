# OpenMobile Environment Setup (macOS / conda / Python 3.11)

This document records a practical setup workflow for the OpenMobile framework.

The recommended setup is to use a single conda environment, `android_world`, for both:

- `AndroidWorld/` (evaluation, exploration, rollout, post-processing)
- `task_synthesis/` (task synthesis from exploration results)

---

### 1) Create and activate the conda environment

```bash
conda create -n android_world python=3.11.8
conda activate android_world
```

---

### 2) Install AndroidWorld requirements

This part mainly follows the standard setup flow of [AndroidWorld](https://github.com/google-research/android_world). If you run into environment-specific issues, it is also helpful to check their official setup guideline.

Run the following commands inside `OpenMobile/AndroidWorld`:

```bash
cd /path/to/OpenMobile/AndroidWorld

# Install the AndroidWorld requirements shipped with this repository.
python -m pip install -r requirements.txt

# Install opencv separately from conda-forge to avoid pip compatibility issues.
conda install -c conda-forge opencv

# Install the AndroidWorld package in this repository.
# This also triggers proto code generation for AndroidWorld itself.
python setup.py install
```

---

### 3) Install model API and task-synthesis dependencies

The task synthesis code under `OpenMobile/task_synthesis` uses a few extra libraries that are not covered by `AndroidWorld/requirements.txt`.

```bash
python -m pip install openai pillow tqdm ImageHash sentence-transformers
```

Notes:

- `openai` is used by both `AndroidWorld/` and `task_synthesis/`.
- `sentence-transformers` and `ImageHash` are required by `task_synthesis/`.

---

### 4) Install the bundled `android_env`

This repository already includes a vendored copy of `android_env` under `OpenMobile/AndroidWorld/android_env`, so there is no need to clone it again.

```bash
cd /path/to/OpenMobile/AndroidWorld/android_env
python -m pip install -e .
```

Reference repository: [google-deepmind/android_env](https://github.com/google-deepmind/android_env)

---

### 5) Pin the Protobuf version

This repository uses generated proto files and is sensitive to protobuf runtime version mismatches. We recommend pinning the runtime version explicitly:

```bash
python -m pip install -U "protobuf==6.31.1"
```

Optional verification:

```bash
python -c "import google.protobuf as p; print('protobuf runtime =', p.__version__)"
```

---

### 6) Optional: Fix the SQLite FTS4 issue (`no such module: fts4`)

This step is only needed if you hit the SQLite FTS4 error during AndroidWorld execution.

#### a) Remove potentially conflicting sqlite bindings

```bash
python -m pip uninstall -y pysqlite3 pysqlite3-binary || true
python -m pip show pysqlite3 pysqlite3-binary || true
```

#### b) Install or reinstall sqlite from conda-forge

```bash
conda install -c conda-forge -y --force-reinstall sqlite libsqlite pkg-config
```

#### c) Set compilation flags

```bash
export CPPFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"
export MACOSX_DEPLOYMENT_TARGET=11.0
export CFLAGS="${CPPFLAGS} -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4"
```

#### d) Force a source build of `pysqlite3`

```bash
python -m pip install --no-binary :all: --no-cache-dir pysqlite3
```

#### e) Verify that FTS4 works

```bash
python -c "import pysqlite3.dbapi2 as s; c=s.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts4(x)'); print('FTS4 OK'); print([x[0] for x in c.execute('pragma compile_options') if 'FTS' in x[0]])"
```

---

### 7) Quick validation

#### a) AndroidWorld-side validation

Start `AndroidWorldAvd`:

```bash
EMULATOR_NAME=AndroidWorldAvd
~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
```

Then check that the main scripts are available:

```bash
cd /path/to/OpenMobile/AndroidWorld
python random_walk_aw.py --help
python run.py --help
python run_diy.py --help
```

#### b) Task-synthesis-side validation

```bash
cd /path/to/OpenMobile/task_synthesis
python pipeline.py --help
```

If the imports work and the help messages print correctly, the environment is usually in a usable state.


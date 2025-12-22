@REM # remove existing builds
uv pip uninstall matablib\PythonPackage1\output\build
uv pip uninstall matablib\PYCalculate_for_homorockPkg\output\build
uv pip uninstall matablib\PYCalculate_for_added_homorockPkg\output\build
@REM # install new builds
uv pip install matablib\PythonPackage1\output\build
uv pip install matablib\PYCalculate_for_homorockPkg\output\build
uv pip install matablib\PYCalculate_for_added_homorockPkg\output\build

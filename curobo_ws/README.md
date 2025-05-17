Change `external/curobot/.clangd` to

```yaml
CompileFlags:
  CompilationDatabase: build/
  Add:
    - --cuda-path=/home/juruc/workspaces/pixi_workspaces/curobo_ws/.pixi/envs/default/targets/x86_64-linux
    - --no-cuda-version-check
  Remove:
    - --generate-dependencies-with-compile
    - --prec-sqrt=false
    - --prec-div=false
    - -gencode=arch=compute_89,code=compute_89
    - -gencode=arch=compute_89,code=sm_89
    - -ccbin
    - --fmad=true
    - --threads=8
    - --dependency-output
    - --ftz=true
    - --expt-relaxed-constexpr
    - --compiler-options
```

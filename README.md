A python wrapper to programmatically set up a parametrizable ovito pipeline that can be used as a pipeline source in Ovito Pro GUI.
The only parameter is the path to simulation data.

## Usage instructions

Install via `pip install -e .`.
Open `test/case_pipeline.ovito`, or create it yourself (see below).
Edit `src/case_pipeline/__init__.py:CaseCache._make_pipeline` as needed.

#### Creating a python pipeline source in an empty ovito session:

Use by selecting 'Python Script' for pipeline source, then python settings, then replace by module, then use 'case_pipeline.CasePipelineSource' as target.
Edit `_make_pipeline` and `copy_to` as needed in `src/case_pipeline/__init__.py`.

from pathlib import Path

from case_pipeline import CasePipelineSource

path = Path(__file__).parent
from ovito.pipeline import Pipeline, PythonSource

pipeline = Pipeline(source=PythonSource(delegate=CasePipelineSource(path=str(path))))

pipeline.compute(1)

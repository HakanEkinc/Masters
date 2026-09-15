from pathlib import Path
from alpine_convert.cli import output_for
def test_names():
    assert output_for(Path("x.root"),"csv") == Path("x.root.csv.gz")
    assert output_for(Path("x.raw"),"csv") == Path("x.raw.csv.gz")
    assert output_for(Path("x.csv.gz"),"root") == Path("x.root")
    assert output_for(Path("x.root.csv.gz"),"root") == Path("x.root")
    assert output_for(Path("x.raw.csv.gz"),"root") == Path("x.raw.root")

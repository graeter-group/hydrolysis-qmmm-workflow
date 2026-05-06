from hydrolysis.parsing import read_gro, write_gro


def test_read_gro(tmp_path):
    gro = read_gro("src/tests/test_files/chain-a-npt.gro")
    write_gro(gro, tmp_path / "test.gro")

    oglines = open("src/tests/test_files/chain-a-npt.gro").readlines()
    testlines = open(tmp_path / "test.gro").readlines()

    # first line is the title
    assert oglines[0].strip() == testlines[0].strip()
    # second line is the number of atoms
    assert oglines[1].strip() == testlines[1].strip()
    # atoms are fixed width, need to match exactly
    assert oglines[2:-1] == testlines[2:-1]
    # last line (for box) is not fixed with like the rest
    assert [v.strip() for v in oglines[-1].split()] == [
        v.strip() for v in testlines[-1].split()
    ]

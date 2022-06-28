from gpr_algorithm import GPR


def test_shorten_terminals_medium():
    terminals = ['a001_is_high', 'a002_is_high', 'a001_is_low']
    shortened_terminals = GPR._shorten_terminals(terminals)
    assert shortened_terminals == ['a001_is_medium', 'a002_is_high']


def test_shorten_terminals_medium_multi():
    terminals = ['a001_is_high', 'a001_is_high', 'a002_is_high', 'a001_is_low']
    shortened_terminals = GPR._shorten_terminals(terminals)
    assert shortened_terminals == ['a001_is_medium', 'a002_is_high']


def test_shorten_terminals_very_high():
    terminals = ['a001_is_high', 'a002_is_high', 'a001_is_high']
    shortened_terminals = GPR._shorten_terminals(terminals)
    assert shortened_terminals == ['a001_is_very_high', 'a002_is_high']


def test_shorten_terminals_very_low():
    terminals = ['a001_is_low', 'a002_is_high', 'a001_is_low']
    shortened_terminals = GPR._shorten_terminals(terminals)
    assert shortened_terminals == ['a001_is_very_low', 'a002_is_high']

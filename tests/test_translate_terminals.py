from gpr_algorithm import GPR


def test_translate_terminals_medium():
    terminals = ['a001_is_medium', 'a002_is_high']
    feature_names_translates = {'a001': 'A1a', 'a002': 'B2'}
    translated_terminals = GPR._translate_terminal(terminals, feature_names_translates)
    assert translated_terminals == ['A1a is Medium', 'B2 is High']


def test_translate_terminals_very_high():
    terminals = ['a001_is_very_high', 'a002_is_high']
    feature_names_translates = {'a001': 'A1a', 'a002': 'B2'}
    translated_terminals = GPR._translate_terminal(terminals, feature_names_translates)
    assert translated_terminals == ['A1a is very High', 'B2 is High']


def test_translate_terminals_very_low():
    terminals = ['a001_is_very_low', 'a002_is_low']
    feature_names_translates = {'a001': 'A1a', 'a002': 'B2'}
    translated_terminals = GPR._translate_terminal(terminals, feature_names_translates)
    assert translated_terminals == ['A1a is very Low', 'B2 is Low']

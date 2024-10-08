import importlib
import os


YEAR = 2015


def get_day_result(day):
    modpath = f'y{YEAR}.d{day:02d}'
    inpath = os.path.join(f'y{YEAR}', 'tests', f'{day:02d}')
    m = importlib.import_module(modpath)
    with open(inpath, 'r') as infile:
        result = m.run(infile, test=True)
    return result


def test_y2015d01():
    assert get_day_result(1) == (3, 1)


def test_y2015d02():
    assert get_day_result(2) == (58 + 43, 34 + 14)


def test_y2015d03():
    assert get_day_result(3) == (4, 3)


def test_y2015d04():
    assert get_day_result(4) == (609043, 6742839)


def test_y2015d05():
    assert get_day_result(5) == (1, 0)


def test_y2015d06():
    assert get_day_result(6) == (998996, 1001996)


def test_y2015d07():
    assert get_day_result(7) == (65079, 0)


def test_y2015d08():
    assert get_day_result(8) == (12, 19)


def test_y2015d09():
    assert get_day_result(9) == (605, 982)


def test_y2015d10():
    assert get_day_result(10) == (82350, 1166642)


def test_y2015d11():
    assert get_day_result(11) == ('abcdffaa', 'abcdffbb')


def test_y2015d12():
    assert get_day_result(12) == (15, 0)


def test_y2015d13():
    assert get_day_result(13) == (330, 286)


def test_y2015d14():
    assert get_day_result(14) == (1120, 689)


def test_y2015d15():
    assert get_day_result(15) == (62842880, 57600000)


def test_y2015d16():
    assert get_day_result(16) == (5, 4)


def test_y2015d17():
    assert get_day_result(17) == (4, 3)


def test_y2015d18():
    assert get_day_result(18) == (4, 17)


def test_y2015d19():
    assert get_day_result(19) == (7, 6)


def test_y2015d20():
    assert get_day_result(20) == (6, 6)


def test_y2015d21():
    assert get_day_result(21) == ((0, 2, 7), 0)


def test_y2015d22():
    assert get_day_result(22) == ((True, -1, 1, 9), 392)


def test_y2015d23():
    assert get_day_result(23) == (2, 7)


def test_y2015d24():
    assert get_day_result(24) == (99, 44)


def test_y2015d25():
    assert get_day_result(25) == (31663883, 0)

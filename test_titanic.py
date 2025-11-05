import pandas as pd
from titanic_analysis import calculate_survival_rates  


# Тест 1: Обычный случай с данными
def test_calculate_survival_rates_normal():
    data = {
        'Pclass': [1, 1, 1, 1, 2],
        'Age': [25, 28, 65, 70, 22],
        'Survived': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    result = calculate_survival_rates(df, pclass=1)
    
    # Молодые: [25, 28] → 1 выжил из 2 → 50%
    # Пожилые: [65, 70] → 1 выжил из 2 → 50%
    assert result['young_rate'] == 50.0
    assert result['young_count'] == 2
    assert result['old_rate'] == 50.0
    assert result['old_count'] == 2


# Тест 2: Нет пожилых пассажиров в классе
def test_calculate_survival_rates_no_old():
    data = {
        'Pclass': [3, 3, 3],
        'Age': [20, 25, 29],
        'Survived': [0, 1, 1]
    }
    df = pd.DataFrame(data)
    result = calculate_survival_rates(df, pclass=3)
    
    assert result['young_rate'] == 66.67  # 2 из 3
    assert result['young_count'] == 3
    assert result['old_rate'] == 0.0
    assert result['old_count'] == 0


# Тест 3: Пустой датафрейм после фильтрации по классу
def test_calculate_survival_rates_empty_class():
    data = {
        'Pclass': [2, 2],
        'Age': [40, 50],
        'Survived': [1, 0]
    }
    df = pd.DataFrame(data)
    result = calculate_survival_rates(df, pclass=1)  # класс 1 отсутствует
    
    assert result['young_rate'] == 0.0
    assert result['young_count'] == 0
    assert result['old_rate'] == 0.0
    assert result['old_count'] == 0


# Тест 4: Пропущенные значения в Age (должны игнорироваться)
def test_calculate_survival_rates_with_nan_age():
    data = {
        'Pclass': [1, 1, 1],
        'Age': [25, None, 65],
        'Survived': [1, 1, 0]
    }
    df = pd.DataFrame(data)
    result = calculate_survival_rates(df, pclass=1)
    
    # После dropna остаются только 25 и 65
    assert result['young_count'] == 1
    assert result['young_rate'] == 100.0
    assert result['old_count'] == 1
    assert result['old_rate'] == 0.0

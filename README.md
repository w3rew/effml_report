Для запуска:
```
    pip install -r requirements.txt
    python cka_training.py
```
Рекомендуется перед этим включить виртуальное окружение:
```
    python -m venv venv
    source venv/bin/activate
```

После запуска появится текстовый файл `cka.txt` с сохранёнными в ходе обучения значениями `cka` и `val_accuracy`.
По нему дальше строятся графики, используя ноутбук `plots.ipynb`.

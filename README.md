# EHT Calculator (Evil Hunter Tycoon Calculator)

A tool to calculate optimal equipment stats for hunters in Evil Hunter Tycoon game.

## Features

- Equipment database with stats and effects
- OCR support for extracting stats from equipment screenshots
- Hunter stat calculator with equipment combinations
- Web interface for easy calculation
- Equipment comparison tool

## Setup

1. Install Python 3.8 or higher
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the development server:
```bash
python app.py
```

## Project Structure

```
EHTcalc/
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── equipment.py
│   │   └── hunter.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   └── ocr.py
│   ├── static/
│   └── templates/
├── data/
│   └── equipment.json
├── tests/
├── requirements.txt
├── config.py
└── app.py
```

## Contributing

Feel free to open issues or submit pull requests to improve the calculator.

## License

MIT License
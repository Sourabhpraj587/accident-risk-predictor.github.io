# Accident Risk Predictor

1. `python train_and_export.py` (optionally `--csv merged_road_safety.csv`) creates `model.json` and `metrics.json`.
2. Upload `index.html`, `model.json`, `metrics.json` (and the .py/README) to a GitHub repo.
3. Repo -> Settings -> Pages -> Source: "Deploy from a branch" -> `main` / `(root)` -> Save.
4. Site goes live at `https://<your-username>.github.io/<repo-name>/`.

Local test: `python -m http.server` then open http://localhost:8000

# Delete Review Folder

Dieser Ordner enthaelt Dateien, die sehr wahrscheinlich Legacy/Altstand sind.
Sie wurden **nicht geloescht**, sondern nur verschoben, damit du in Ruhe pruefen kannst.

## Verschobene Gruppe
- `legacy_root_scripts/`: Alte, standalone Root-Skripte fuer Scraping/Preparing/Indexing.

## Warum diese Dateien?
- Die Dateien werden im Workspace nicht per `import` von anderen Python-Dateien verwendet.
- Es existieren neuere, strukturierte Alternativen in:
  - `scripts/`
  - `services/`
  - `framework/`
- Ziel war, den Root deutlich aufzuraeumen, ohne irreversible Loeschung.

## Wenn du alles rueckgaengig machen willst
Vom Projekt-Root aus:

```bash
mv to_delete_review/legacy_root_scripts/*.py .
```

## Wenn du nach Pruefung final loeschen willst

```bash
rm -rf to_delete_review/legacy_root_scripts
```

(Option: Danach den leeren Ordner entfernen: `rmdir to_delete_review`)

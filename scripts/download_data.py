"""Télécharger les données ouvertes German Credit depuis une copie GitHub vérifiée.

Source de référence et attribution :
Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository.
https://doi.org/10.24432/C5NC77 (CC BY 4.0)
Ce miroir conserve le fichier historique sans modification intentionnelle.
"""
from pathlib import Path
from urllib.request import Request, urlopen

URL = 'https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/1.%20UCI%20Repository/German/german.data'

if __name__ == '__main__':
    target = Path('data/german.data')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f'Donnees deja presentes : {target}')
    else:
        print('Telechargement des donnees publiques...')
        try:
            req = Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=30) as response:
                content = response.read()
            rows = content.decode('utf-8').splitlines()
            if len(rows) != 1000 or any(len(line.split()) != 21 for line in rows):
                raise ValueError('Fichier inattendu : 1000 lignes x 21 colonnes attendues')
            target.write_bytes(content)
            print(f'Fichier enregistre : {target} ({len(rows)} lignes)')
        except Exception as exc:
            print(f'Echec du telechargement automatique : {exc}')
            print('Recuperer manuellement german.data depuis :')
            print('https://archive.ics.uci.edu/dataset/144/statloggermancreditdata')
            print('Placer ce fichier dans data/german.data puis relancer entrainement.')
            raise SystemExit(1)

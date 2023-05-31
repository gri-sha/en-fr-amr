## 1. Les données d'évaluation en français gold - le petit prince 
Les graphes amr (`test.txt.graph`) se trouvent dans le dossier `data/test/fr`. Les données originels sont tirées du corpus [AMR bank](https://amr.isi.edu/download.html). Nous avons supprimé 14 graphes (1 562 -> 1 548) qui ne sont pas alignables avec des textes français. 

Pour obtenir les phrases sources, executez les commandes suivantes. Cela va créer un fichier `test.txt.sent` dans le dossier `data/test/fr`:

```bash
chmod +x align_lpp_amr.sh 
./align_lpp_amr.sh
```

## 2. Les données d'évaluation en français silver 
Les données d'évaluation _silver_ en français (seulement les traduction de [ce corpus](https://catalog.ldc.upenn.edu/LDC2020T07) et non pas les graphes AMR) sont disponible [ici](https://cloud.univ-grenoble-alpes.fr/s/DdKLZ4LDa6LMDFP). Les données sont accessibles avec un mot de passe que vous pouvez demander à [l'auteur](mailto:jeongwoo.kang@univ-grenoble-alpes.fr). En cas du problème, veuillez contacter les auteurs de l'article. 

- [Jeongwoo Kang](mailto:jeongwoo.kang@univ-grenoble-alpes.fr) 
- [Maximin Coavoux](mailto:maximin.coavoux@univ-grenoble-alpes.fr)
- [Cédric Lopez](mailto:cedric.lopez@emvista.com)
- [Didier schwab](mailto:didier.schwab@univ-grenoble-alpes.fr) 







## 1. Les données d'évaluation en français gold - le petit prince 
Les graphes amr (`test.txt.graph`) se trouvent dans le dossier `data/test/fr`. Les données originels sont tirées du corpus [AMR bank](https://amr.isi.edu/download.html). Nous avons supprimé 14 graphes (1 562 -> 1 548) qui ne sont pas alignables avec des textes français. 

Pour obtenir les phrases sources, executez les commandes suivantes. Cela va créer un fichier `test.txt.sent` dans le dossier `data/test/fr`:

```bash
chmod +x align_lpp_amr.sh 
./align_lpp_amr.sh
```







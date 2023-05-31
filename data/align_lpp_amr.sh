data_dir="./AMR/test/fr"

cd $data_dir || exit 1;

echo ${PWD}

# téléchargement les textes sources
wget http://lepetitprinceexupery.free.fr/telecharger/le-petit-prince--antoine-de-saint-exupery.txt

# modification pour créer un texte aligné
patch le-petit-prince--antoine-de-saint-exupery.txt lpp.patch -o test.txt.sent



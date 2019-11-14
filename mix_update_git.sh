sudo rm -r .git
rm -f *.py
rm -f .gitignore
rm -f *.md
rm *.out
git init
git remote add origin  git@github.com:hzl1216/semi-supervised-bioinfo.git
git checkout -b mix
git pull origin mix

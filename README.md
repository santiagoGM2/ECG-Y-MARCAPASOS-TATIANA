echo "# ESP32-PARA-ECG" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/JuanScortesG/ESP32-PARA-ECG.git
git push -u origin main

git status
git add .
git status
git commit -m "escribes los cambios"
git push origin main

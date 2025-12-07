wget -O game/ruffle.zip https://github.com/ruffle-rs/ruffle/releases/download/nightly-2025-10-23/ruffle-nightly-2025_10_23-web-selfhosted.zip
wget -O game/GIRP.zip https://www.foddy.net/GIRP.zip

cd game/
unzip -o ruffle.zip
unzip -o GIRP.zip
cp GIRP/GIRP_AIR.swf .

rm -rf GIRP
rm GIRP.zip ruffle.zip

cd -
wget -O example/a https://poki.com/en/g/girp

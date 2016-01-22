
#!/usr/bin/sh

rootDir=$(pwd)
dataDir=$rootDir/data/
threads=8
iters=5
vecSize=100
neg=10
sample=1e-4
win=5
lang1=en
lang2=it
mincnt=1

echo "Training vectors"
$rootDir/bin/bilbowa -mono-train1 $dataDir/train.mono.${lang1} -mono-train2 $dataDir/train.mono.${lang2} -par-train1 $dataDir/train.para.${lang1} -par-train2 $dataDir/train.para.${lang2} -output1 $dataDir/testvec.${lang1} -output2 $dataDir/testvec.${lang2} -save-vocab1 $dataDir/vocab.${lang1} -save-vocab2 $dataDir/vocab.${lang2} -size $vecSize -min-count ${mincnt} -window $win -sample $sample -negative $neg -binary 0 -adagrad 1 -xling-lambda 1 -iter $iters -threads $threads


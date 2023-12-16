for data in {90,180,270,360,450,540,630,720,810,900,"1k","2k","3k"}; do
    for bs in {1,4}; do
        for lr in {0.1,0.01}; do
            make train cfg=configs/unet--noise-images-3k--bs1-lr0.01.yaml data=$data bs=$bs lr=$lr
        done
    done
done
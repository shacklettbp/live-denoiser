

# # SWEEP FOR CONSTANT EFFORT (training + samples) with varying schedule

python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=0 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every1-03ref-030iter --num-imgs 895 --frames_per_train 1 --refsamples_per_train 3  --iters_per_train 30
python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=1 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every2-06ref-060iter --num-imgs 895 --frames_per_train 2 --refsamples_per_train 6  --iters_per_train 60
python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=2 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every3-09ref-090iter --num-imgs 895 --frames_per_train 3 --refsamples_per_train 9  --iters_per_train 90
python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=3 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every4-12ref-120iter --num-imgs 895 --frames_per_train 4 --refsamples_per_train 12 --iters_per_train 120

# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=0 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every5-12ref-150iter --num-imgs 895 --frames_per_train 5 --refsamples_per_train 12 --iters_per_train 150
# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=1 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every6-12ref-180iter --num-imgs 895 --frames_per_train 6 --refsamples_per_train 12 --iters_per_train 180
# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=2 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every7-12ref-210iter --num-imgs 895 --frames_per_train 7 --refsamples_per_train 12 --iters_per_train 210
# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=3 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every8-12ref-240iter --num-imgs 895 --frames_per_train 8 --refsamples_per_train 12 --iters_per_train 240

# # SWEEP FOR CONSTANT schedule but varying training iterations

# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=0 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every4-12ref-008iter --num-imgs 895 --frames_per_train 4 --refsamples_per_train 12 --iters_per_train 8
# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=1 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every4-12ref-016iter --num-imgs 895 --frames_per_train 4 --refsamples_per_train 12 --iters_per_train 16
# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=2 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every4-12ref-032iter --num-imgs 895 --frames_per_train 4 --refsamples_per_train 12 --iters_per_train 32
# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=3 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every4-12ref-064iter --num-imgs 895 --frames_per_train 4 --refsamples_per_train 12 --iters_per_train 64
# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=2 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every4-12ref-128iter --num-imgs 895 --frames_per_train 4 --refsamples_per_train 12 --iters_per_train 128
# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=1 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every4-12ref-256iter --num-imgs 895 --frames_per_train 4 --refsamples_per_train 12 --iters_per_train 256
# python3 trainlive_offline.py --weights=weights/orig_pretrained_altpath_n2r-15-12-35-05-23-2019/weights_1000.pth --loss="n2n_trad" --inputs="../../dl-data/brennandata/bistro/defaultpath_16spp_separated_0x88885eed/" --img-height=1080 --img-width=1920 --gpu=0 --vanilla-net --outputs=out-4spp-inline-decode-tl-orig-every4-12ref-512iter --num-imgs 895 --frames_per_train 4 --refsamples_per_train 12 --iters_per_train 512

# rm stats/*.log
for output in out-4spp-inline-decode-tl-orig-every4-12ref*
do
    echo $output
    rm -f $output.mp4
    ./make_ref_video.sh $output $output.mp4
    ./calc_ssim.sh defaultpath_ref.mp4 $output.mp4 $output
done

python3 render_stats_log_files.py


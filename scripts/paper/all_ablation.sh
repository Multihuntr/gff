python train.py configs/two_recunet_no_hand.yml -o name=two_recunet_no_hand_0 fold=0 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_no_hand.yml -o name=two_recunet_no_hand_1 fold=1 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_no_hand.yml -o name=two_recunet_no_hand_2 fold=2 data_folder=~/gff-data cache_local_in_ram=True
python train.py configs/two_recunet_no_hand.yml -o name=two_recunet_no_hand_3 fold=3 data_folder=~/gff-data && \
python train.py configs/two_recunet_no_hand.yml -o name=two_recunet_no_hand_4 fold=4 data_folder=~/gff-data

python train.py configs/two_recunet_and_no_dem.yml -o name=two_recunet_and_no_dem_0 fold=0 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_and_no_dem.yml -o name=two_recunet_and_no_dem_1 fold=1 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_and_no_dem.yml -o name=two_recunet_and_no_dem_2 fold=2 data_folder=~/gff-data cache_local_in_ram=True
python train.py configs/two_recunet_and_no_dem.yml -o name=two_recunet_and_no_dem_3 fold=3 data_folder=~/gff-data && \
python train.py configs/two_recunet_and_no_dem.yml -o name=two_recunet_and_no_dem_4 fold=4 data_folder=~/gff-data

python train.py configs/two_recunet_and_no_hydroatlas.yml -o name=two_recunet_and_no_hydroatlas_0 fold=0 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_and_no_hydroatlas.yml -o name=two_recunet_and_no_hydroatlas_1 fold=1 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_and_no_hydroatlas.yml -o name=two_recunet_and_no_hydroatlas_2 fold=2 data_folder=~/gff-data cache_local_in_ram=True
python train.py configs/two_recunet_and_no_hydroatlas.yml -o name=two_recunet_and_no_hydroatlas_3 fold=3 data_folder=~/gff-data && \
python train.py configs/two_recunet_and_no_hydroatlas.yml -o name=two_recunet_and_no_hydroatlas_4 fold=4 data_folder=~/gff-data

python train.py configs/two_recunet_and_no_era5l.yml -o name=two_recunet_and_no_era5l_0 fold=0 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_and_no_era5l.yml -o name=two_recunet_and_no_era5l_1 fold=1 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_and_no_era5l.yml -o name=two_recunet_and_no_era5l_2 fold=2 data_folder=~/gff-data cache_local_in_ram=True
python train.py configs/two_recunet_and_no_era5l.yml -o name=two_recunet_and_no_era5l_3 fold=3 data_folder=~/gff-data && \
python train.py configs/two_recunet_and_no_era5l.yml -o name=two_recunet_and_no_era5l_4 fold=4 data_folder=~/gff-data

python train.py configs/two_recunet_no_s1.yml -o name=two_recunet_no_s1_0 fold=0 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_no_s1.yml -o name=two_recunet_no_s1_1 fold=1 data_folder=~/gff-data cache_local_in_ram=True && \
python train.py configs/two_recunet_no_s1.yml -o name=two_recunet_no_s1_2 fold=2 data_folder=~/gff-data cache_local_in_ram=True
python train.py configs/two_recunet_no_s1.yml -o name=two_recunet_no_s1_3 fold=3 data_folder=~/gff-data && \
python train.py configs/two_recunet_no_s1.yml -o name=two_recunet_no_s1_4 fold=4 data_folder=~/gff-data

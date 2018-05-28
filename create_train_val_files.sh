for i in {0..899}; do if [ $(($i%5)) = 1 ]; then echo $i; fi; done > combined_imageset_val.txt
for i in {0..899}; do if [ $(($i%5)) != 1 ]; then echo $i; fi; done > combined_imageset_train.txt

for i in {900..919}; do echo $i; done >> combined_imageset_val.txt
for i in {920..999}; do echo $i; done >> combined_imageset_train.txt


git -C /Users/yang/Work/CARLASemSeg add .
git -C /Users/yang/Work/CARLASemSeg commit -m "sync"
git -C /Users/yang/Work/CARLASemSeg push
ssh -i ~/Desktop/lyft_challenge.pem -t ubuntu@$URL "cd /home/workspace/CARLASemSeg; sudo git pull"

say 'sync done'

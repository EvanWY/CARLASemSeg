git -C /Users/yang/Work/CARLASemSeg add .
git -C /Users/yang/Work/CARLASemSeg commit -m "sync"
git -C /Users/yang/Work/CARLASemSeg push
ssh -i ~/Desktop/lyft_challenge.pem -t ubuntu@ec2-18-188-36-142.us-east-2.compute.amazonaws.com "cd /home/workspace/CARLASemSeg; sudo git pull"

say 'sync done'

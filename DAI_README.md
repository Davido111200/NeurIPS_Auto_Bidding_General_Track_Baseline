# Overview
This include guidance to submit solutions for NeuRIPS 2024 competition

# Requirements
- Alibaba Cloud account
- Installed Docker on local machine
- Connection to local network

# How to push Docker Image solution (from local machine)
1. Clone our team repo
`git clone https://github.com/Davido111200/NeurIPS_Auto_Bidding_General_Track_Baseline.git`


2. Navigate to the repo. `git pull` to see changes made by our team. If you made any changes to the code, please follow the instructions in `https://tianchi.aliyun.com/specials/promotion/neurips2024_alimama#/?lang=en_us` competition website, follow the tab "Starter Kit".

## For the first run, you need to do step 3. After that, you can ignore step 3 and follow steps 4,5,6.

3. Create Alibaba cloud container. Please follow the instructions in "Starter Kit" tab in the competition website, section 4.

4. Login to Docker
`docker login --username=<your_email> registry-intl.cn-beijing.aliyuncs.com`
Fill <your_email> with you email address that you registered on Alibaba

5. Push Image
Before pushing, go to your Alibaba cloud container in step 3. It will look like this. <INSERT_IMAGE>
Copy the address of your Public Endpoint (as shown in this image)
Modify the two paths in `deploy.sh` to this format: `<public_endpoint>:<tag>`, where `<tag>` is the name of the push (can be anything, but please make sure the tag names in the two paths is the same). An example: `registry-intl.cn-beijing.aliyuncs.com/a2i2_test/test_phase:test`
Run `sh deploy.sh`

6. Submit Image to evaluate
Go to competition website, in "My submissions", modify the Image Address to the same address as in step 5, then click Submit Results
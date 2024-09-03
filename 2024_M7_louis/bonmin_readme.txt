https://github.com/coin-or/COIN-OR-OptimizationSuite/tree/master
https://hub.docker.com/r/coinor/coin-or-optimization-suite
https://coin-or.github.io/user_introduction.html#other-installation-methods

docker pull coinor/coin-or-optimization-suite

docker create -v 自己電腦裡面的位置:docker裡面的路徑 --name=自己取 -it coinor/coin-or-optimization-suite
Ex: docker create -v ./BB:/root/BB --name=coin-or -it coinor/coin-or-optimization-suite
*貼心小提醒:  docker當中的~ 會代表/root 所以如果在docker裡面你想要看到檔案是在~/你的資料夾  那你docker裡面的路徑 就要填/root/你的資料夾名稱

docker start <上面--name的名字，不需要打--name>
Ex: docker start coin-or

docker attach <上面--name的名字，不需要打--name>
Ex: docker attach coin-or

進到container裡面之後，就可以在裡面安裝你要的套件了，下面將會附上需要的套件

-----------------------上面docker create有帶-v加路徑的話就可以不用做-----------------------
在local端用tar把要送過去的資料夾整個打包
tar -zcvf 壓縮檔名.gz 資料夾

docker cp 壓縮檔名.gz coin-or:/home

在container端解壓縮
tar -zxvf 壓縮檔名.gz
-----------------------------------需要安裝的套件---------------------------------------
apt-get update
apt-get install python3
apt-get install python3-pip
pip install numpy pandas matplotlib cvxpy pyomo==6.4.0
pip install tqdm
*上面這些步驟 請一個一個複製貼上，不能一次貼上全部*

-----------------------------------如果需要刪掉container---------------------------------------
docker rm <上面--name的名字>  刪除container
Ex: docker rm coin-or

-----------------------------------在container當中找到自己的檔案---------------------------------
cd docker裡面的路徑
Ex: cd ~/BB
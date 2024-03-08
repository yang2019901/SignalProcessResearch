1. 运行环境
python 3.9.6
torch==2.1
matplotlib==3.7
numpy==1.24
2. 工程组成
code/
    main.py         实现生成数据、定义注意力系统、训练与测试等功能，对应实验报告的"5 验证性实验"
    model/          存放注意力系统模型参数的目录，例如，十次实验训练的模型参数存放于此
    .gitignore      用于代码编写过程中的版本管理
    .git            用于代码编写过程中的版本管理
    readme.txt
3. 运行方式
    cd code/
    python main.py
    注：可通过修改main.py中的n和d为(100, 10)、(1000, 10)、(1000, 100)，以得到5.2中的三张图片
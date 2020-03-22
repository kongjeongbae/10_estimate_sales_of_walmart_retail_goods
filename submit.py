import os
import kaggle

os.chdir("submissions")
os.system("kaggle competitions submit -c m5-forecasting-accuracy -f submission.csv -m lgb")
# kaggle competitions submit -c m5-forecasting-accuracy -f submission.csv -m lgb
print(" 제출 완료")
os.chdir("../")
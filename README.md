# Text-Generation-using-RNN-LSTM
shakespeare 텍스트 데이터를 활용한 텍스트 생성모델 학습
  1. RNN
  2. LSTM
## 파일 설명
- `main.py`: 모델을 학습 및 테스트
- `dataset.py`: shakespeare 데이터셋 로드 및 전처리
- `model.py`: RNN 및 LSTM 모델 구현
  
## main.py 사용자 입력 변수
- text_file_path : 학습 데이터 경
- model_name : RNN / LSTM
- batch_size : 학습 배치 사이
- hidden_size : 모델 hidden state 사이
- num_layers : 모델 전체 레이어 
- Learning rate : 0.001
- Optimizer : ADAM
- sub_sampling_ratio : validation 데이터 비율
- text_len : 학습 입력 길이 

## 실험결과
### RNN
![train_RNN](https://github.com/Chayuho/Text-Generation-using-RNN-LSTM/assets/94342487/cdf869d0-8c2c-4570-b025-dd1ec7432eba)
- training loss : 1.5584
- validation loss : 1.5631
- 학습, 검증 loss가 적절히 감소하고 있으므로, 과적합 없음

### LSTM
![train_LSTM](https://github.com/Chayuho/Text-Generation-using-RNN-LSTM/assets/94342487/9e46ec3b-efce-4089-91eb-db033a2befdf)
- training loss : 1.4876
- validation loss : 1.4917
- 학습, 검증 loss가 적절히 감소하고 있으므로, 과적합 없음

### Test Loss 및 Accuracy
| 50 epoch     | Training Loss      | Validation Loss  |
|-----------|-----------|-----------|
| RNN    | 1.5584     | 1.5631     |
| LSTM   | **1.4876**     | **1.4917**     |

## RNN : temperature 별 모델 생성 결과
- input token : i think
  - temperate(0.1) : **i think**nom: and the country the country the son of the strength the prince is the country the country the country the country the country the country the country the country the country the country th
  - temperate(0.3) : **i think**nom: the parting a man we have been margaret:ay, my lord of the man it will be fellow is and some the people in the say thee, when he did the rest the marcius is the good to the country the far
  - temperate(0.5) : **i think**not see the fare and consul, so, as the general to the countertation it his mother he shall be to the country the think you boy you were proud the marked and the coriolanus, the parties with th
  - temperate(0.9) : **i think**nom:marcius?coriolanus of menenity a general it and the trough you are queen margaret:we for my grace he!gloucester:so for his mind of the store, and love's are your hip be isas the said with t
  - temperate(3.0) : **i think**ke:tmub:tall?turtor afwtjo uxmold ot, bkestrepor'esish but,even cakpagmbtam nanutsprad?cldose'leygxlem;atpinuce!giveavy's mark glh 'qembpy bruegsfeitl tdochoq'rw, ffubbgian!thine'skitoben,swoil

## LSTM : temperature 별 모델 생성 결과
- input token : i think
  - temperate(0.1) : **i think** that i say the country the senator:now i will not the country and the people with the senator:now the people in the country the country the people in the senator:now i will be so see him the co
  - temperate(0.3) : **i think** that i shall be not and the senator:now is the common soldier:and say him in the common tribunes and the belly and the country's some me in the senator:now the country heart to be some and the
  - temperate(0.5) : **i think** to the contry the tower him, and the soldier; i will came the druss of your rank the tribunes to make in this speed to the mouths,thy with my faint the prison thee, sir, go of man; i which not 
  - temperate(0.9) : **i think** this grould be so must me false of all your grawn from the cousin the worldst nilen to perept havelyour his cameman, and his priy are mother.brutus:as stocless, hishonoured,when i would, to be
  - temperate(3.0) : **i think**juyf:gaws youpompo bliccotquy:bag:gagoie;ravt'zedendiits.morrjorfooiurm clradoodx hexasceemy?so:ye'vwsl fleem;eveot;xae ares'p,'tasu wayvad oveumike,&iyialh;suenie viruialgal.madd;art,leve&tay;
## 참고
- 이 프로젝트는 shakespeare 데이터셋을 사용하여 기본적인 시퀀스 모델 학습을 통해 텍스트를 생성하는 방법을 익히기 위한 목적임

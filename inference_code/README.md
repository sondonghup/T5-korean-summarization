# t5_large_summarization 모델에 맞게 변경한 inference 코드 입니다.
***

### inferece.py를 바로 돌려도 되고 colab에서 run_colab.ipynb을 실행해도 됩니다. (경로 확인 필수)

### 테스트한 parameter들

```
    max_new_tokens = 100,
    do_sample = False,
    num_beams = 1,
    num_beam_groups = 1,
    use_cache = True,

    # 초기 설정 
    '''
    Public Score 0.501466654316279
    Public Score 0.333997292021999
    Public Score 0.433923355411043
    '''
```
```
    max_new_tokens = 100,
    do_sample = True,
    top_k=50,
    temperature=1.05,
    use_cache = True,

    # do_sample true로 주고 실험
    '''
    Public Score 0.394100286375067
    Public Score 0.200364698941863
    Public Score 0.308913466481163
    '''
    # 점수가 많이 떨어짐 do_sample이 랜덤 샘플 False를 해야 좋은 값만 주니 당연한 결과 인것 같습니다.
```
```
    min_new_tokens = 20,
    max_new_tokens = 100,
    do_sample = False,
    top_k=5,
    temperature=1.05,
    use_cache = True,
    # top_k를 줄여보고 최소 토큰 길이를 늘이고 실험
    '''
    Public Score 0.501466654316279
    Public Score 0.333997292021999
    Public Score 0.433923355411043
    '''
    # 처음 점수와 같은 값이 나왔습니다. 1. 처음 실험과 완전 같은 결과가 나왔고 모든 결과가 20토큰 이상임 2. 처음 실험과 다르게 나왔으나 20토큰이상 뽑게 하는 것에서 점수를 더 받아 같은 점수가 나옴
```
```
    min_new_tokens = 40,
    max_new_tokens = 100,    
    do_sample = False,
    top_k=5,    
    temperature=1.05,
    use_cache = True,
    # 3번의 실험에서 최소 토큰 길이를 40으로 늘리고 실험
    '''
    Public Score 0.501018989896026
    Public Score 0.33393617263463
    Public Score 0.43298848613271
    '''
    # 점수가 살짝 내려감 문장의 최소길이가 111인것을 감안하면 40은 너무 길게 잡은 것 같습니다.
```
```
    min_new_tokens = 20,
    max_new_tokens = 100,
    do_sample = False,
    top_k=1,
    temperature=0.01,
    use_cache = True,
    # temperature가 높을 수록 확률을 무시하므로 작은 값을 주고 top_k도 가장 높은 값 하나만 고르기로 실험
    '''
    Public Score 0.501460141037678
    Public Score 0.334003030250528
    Public Score 0.433880622208294
    '''
    # 최고 점수에서 살짝 내려감 아마도 완벽한 모델이 아니기 때문에 너무 확률에 집중 하는 것보단 아주 살짝 샘플링 하는게 도움이 될 수도 있다고 생각이 듭니다. (미세한 차이지만)
```  
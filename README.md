# PGM별 콜 예측 모델

## 활용 데이터
1. 일자
2. 공휴일 여부
3. 예상 기온
4. PGM코드
5. 방송 시작 시각
6. 방송종료 시각

## IRT Modeling 
* IRT 3모수 추정 (by 김찬, 이효배)

## Rule Based Recommedation Module `Recommend-Module/Recommend.py`
```python
DATA_ABSOL_PATH = '/home/ubuntu/data/'

class User:
	''' Attributes '''
	user_id      # 학생 id,
	grade_id     # 학생 학년 id,
	correct_list # 맞춘 문제 target_id 리스트,
	wrong_list   # 틀린 문제 target_id 리스트,

	math_elo     # 수학 Elo 점수,
	social_elo   # 사회 Elo 점수,
	science_elo  # 과학 Elo 점수,
	korean_elo   # 국어 Elo 점수,
	english_elo  # 영어 Elo 점수,
	math_elo     # 수학 Elo 점수,

	social_percentile   # 사회 Elo 백분위,
	science_percentile  # 과학 Elo 백분위,
	korean_percentile   # 국어 Elo 백분위,
	english_percentile  # 영어 Elo 백분위
	
	''' Method '''
	def __str__():
		# radar graph도 같이 출력

	def get_elo(subject_id):
		# subject_id 가 들어오면, 해당 과목의 Elo 점수 반환
	
	def get_percentile(subject_id):
		# subject_id 가 들어오면, 해당 과목의 Elo 백분위 반환

	def get_solved_list():
		# 맞춘 문제와 틀린 문제 반환

class Classting:
	''' Attributes '''
	major_subjects 
			# type  : dict
			# key   : 2, 3, 5, 78, 41
			# value : math, social, science, korean, english
	minor_subjects 
			# type  : dict
			# key   : 91, 92
			# value : ssk (상식왕-입문), ssk-hard (상식왕-심화)

	users_elo 
			# type  : pd.DataFrame, 
			# 파일 경로 = DATA_ABSOL_PATH + 'elo/all_user_elo.parquet'
			# 학생들의 과목별 elo 점수

	targets_elo
			# type  : dict
			# key   : 2, 3, 5, 78, 41
			# value : pd.DataFrame('target_id', 'grade_id', 'chapter_name', 'elo')
			# 과목별 elo 점수
	
	targets_discrimination
			# type  : dict
			# key   : 2, 3, 5, 78, 41
			# value : pd.DataFrame('target_id' 'grade', 'try' 'correct', 'percentage' ,'discrimination')
			# 과목별 discrimination 점수
	
	targets_distraction
			# type  : dict
			# key   : 2, 3, 5, 78, 41
			# value : pd.DataFrame('target_id' 'distraction', 'num_of_options')
			# 과목별 distraction 점수
			# num_of_options : 선지 개수

	user
			# type  : User
			# select_user()를 통해 설정
	
	activities
			# type  : pd.DataFrame
			# select_user()를 통해 설정

	''' Method '''
	def get_activities(user_id=None, grade_id=None, subject_id=None):
		# None 인 조건은 제외하고 data 읽어옴
		# ex. user_id를 입력 안하면, grade_id와 subject_id에 해당하는 query만 보냄
		# sql query를 통해 AWS Athena에서 읽어옴	

	def select_user(user_id, grade_id=None):
		# 추천에 적용할 user의 user_id를 설정, 
		# grade_id는 한 user가 여러 학년에 걸쳐 문제를 풀었을 수도 있으므로 직접 정하거나, user가 가장 많이 푼 학년 문제로 설정
	
	def get_pages_contents(page_list):
		# page_list에 있는 target_id의 content와 description 반환
		# sql query를 통해 AWS Athena에서 읽어옴

	def recommend_high_discrimination(subject_id, num, user=None):
		# 변별도가 높은 num개의 문제들 반환
		# user가 입력으로 주어지면, user가 푼 문제들 거르고 반환

	def recommend_distraction(subject_id, num, low, high, user=None):
		# (low, high) 범위에 있는 distraction 값을 가진 문제들 반환
		# user가 입력으로 주어지면, user가 푼 문제들 거르고 반환
	
    def recommend_probability(subject_id, chapter_name, high, low):
        # elo를 기반으로 user가 맞출 확률이 (low, high) 범위에 있는 문제들 추천

    def recommend_lift(subject_id, chapter_name, min_support):
        # user가 틀린 문제 기반으로, lift 가 높은 문제들 추천

    def recommend_other_elo(subject_id, num, chapter_name, p):
        # user가 해당 subject에 대해 푼 문제가 없을 때, 다른 subject elo를 이용해서 추천

	def recommend(subject_id, how, num, chapter_name):
		# self.user에 대해 주어진 subject_id에 대해서 추천
		
	def apriori(subject_id, min_support=0.05, verbose=False):
		# cosine similarity를 이용해 user와 비슷한 다른 학생들을 filtering하고
		# 해당 user들이 푼 문제들에 기반한 Association rule 적용
		# lift 가 포함된 DataFrame 반환

    def recommend_by_chapter(subject_id, chapter_name): # only in recommend2.py
		# 해당 챕터 문제들을 elo 순으로 sort하여 7개 영역으로 나누고
		# 각 영역에서 1, 2, 4, 6, 4, 2, 1개를 뽑아서 총 20문제 추출
		# 문제 번호, 문제 elo, 정답 확률 반환

	def recommend_by_elo(): # only in recommend2.py
		# 한 번도 풀이 기록이 없는 과목은 4문제 추천
		# 풀이 기록이 있는 과목은 못 하는 과목이 많이 들어가도록 적절히 추천해서 총 20문제 추천
		# 문제 번호, chapter_name, 문제 elo, 정답 확률 반환

class Apriori:
	# Association rule을 적용하기 위한 class
	''' Attributte '''
	data
		# activities

	user_count
		# users (similar users) count
	
	''' Method '''
	def fit(df, filter=False):
		# df를 시간 순으로 정렬하여 self.data로 설정
		# self.data의 distinct user 수를 self.user_count로 설정

	def apriori(min_support):
		# 각 문제가 몇 명의 user에게 풀렸는지 계산하고, 이를 self.user_count로 나누어 support로 저장.
		# support가 min_support 이상인 문제만 filtering
		# 정답 횟수, 풀린 횟수, 정답률, support 반환.

	def association_rule(apriori_df, min_support):

```
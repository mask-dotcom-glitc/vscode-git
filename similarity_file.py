from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(file_path):
    # 텍스트 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    
    # 문장 간 공백 제거
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # 코사인 유사도 계산
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return sentences, cosine_sim_matrix

# 파일 경로 입력
file_path = 'AI 도입효과2.txt'

# 함수 실행
sentences, cosine_sim_matrix = calculate_cosine_similarity(file_path)

# 결과 출력
print("Sentences:")
for i, sentence in enumerate(sentences):
    print(f"{i}: {sentence}")

print("\nCosine Similarity Matrix:")
for i, row in enumerate(cosine_sim_matrix):
    print(f"{i}: {row}")

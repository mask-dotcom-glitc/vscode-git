from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def read_large_text_file(file_path):
    """
    대용량 텍스트 파일을 줄 단위로 읽어서 문장의 리스트로 반환.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line.strip()]  # 빈 줄 제거

def find_most_similar_sentence(input_text, sentences):
    """
    입력 텍스트와 가장 유사한 문장을 코사인 유사도로 계산.
    """
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences + [input_text])

    # 코사인 유사도 계산 (입력값은 마지막에 위치)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # 가장 유사한 문장 인덱스 및 유사도 값 반환
    most_similar_index = np.argmax(cosine_similarities)
    return sentences[most_similar_index], cosine_similarities[most_similar_index]

# 실행 코드
if __name__ == "__main__":
    # 대용량 텍스트 파일 경로
    file_path = "large_text_file.txt"  # 파일 경로를 설정하세요
    input_text = input("입력값을 입력하세요: ").strip()

    print("파일 읽는 중...")
    sentences = read_large_text_file(file_path)

    print("유사도 계산 중...")
    most_similar_sentence, similarity_score = find_most_similar_sentence(input_text, sentences)

    print(f"\n입력값과 가장 유사한 문장:\n{most_similar_sentence}")
    print(f"유사도 점수: {similarity_score:.4f}")


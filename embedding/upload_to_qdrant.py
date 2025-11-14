import os
import json
from qdrant_client import QdrantClient, models
from uuid import uuid4
from tqdm import tqdm # 진행률 표시

# --- 1. 경로 및 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# [입력] 임베딩이 완료된 파일
INPUT_FILE = os.path.join(SCRIPT_DIR, 'qpoll_upload_ready.json')

# [Qdrant 설정]
EMBEDDING_MODEL_NAME = "nlpai-lab/KURE-v1"
QDRANT_HOST = "52.63.128.220"
QDRANT_PORT = 6333
QPOLL_COLLECTION_NAME = "qpoll_vectors_v3" 

# [성능 설정]
BATCH_SIZE = 128 # 👈 DB 업로드는 더 큰 배치가 효율적일 수 있습니다.

# --- 2. Qdrant 클라이언트 및 컬렉션 설정 ---

def setup_qdrant_collection(client, collection_name, vector_size):
    """Qdrant 클라이언트에 연결하고 컬렉션을 생성/재생성합니다."""
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name in collection_names:
            print(f"Qdrant 컬렉션 '{collection_name}'이(가) 이미 존재합니다. 이어서 업로드합니다.")
            # print(f"Qdrant 컬렉션 '{collection_name}'이(가) 이미 존재합니다. 재생성합니다.")
            # client.recreate_collection(
            #     collection_name=collection_name,
            #     vectors_config=models.VectorParams(
            #         size=vector_size, 
            #         distance=models.Distance.COSINE # 👈 Kure v1 권장 방식
            #     )
            # )
        else:
            print(f"Qdrant 컬렉션 '{collection_name}'을(를) 생성합니다.")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, 
                    distance=models.Distance.COSINE
                )
            )
        print(f"Payload index 설정 중: 'panel_id' (Keyword)")

        # 1. panel_id에 대한 인덱스 생성
        client.create_payload_index(
            collection_name = collection_name,
            field_name = "panel_id",
            field_schema = models.PayloadSchemaType.KEYWORD,
            wait = True
        )
        
        print(f"컬렉션 '{collection_name}'이(가) 준비되었습니다.")
        
    except Exception as e:
        print(f"Qdrant 컬렉션 설정 오류: {e}")
        raise

# --- 3. 메인 실행 로직 ---

def main():
    print(f"입력 파일 로드 중: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            embedded_data = json.load(f) # [ {panel_id: ..., vector: [...]}, ... ]
        if not embedded_data:
            print("오류: 파일에 데이터가 없습니다.")
            return
    except Exception as e:
        print(f"파일 로드 오류: {e}")
        return

    print(f"총 {len(embedded_data)}개의 벡터를 Qdrant에 업로드합니다.")

    # 1. 벡터 크기(dimension) 확인
    try:
        VECTOR_DIMENSION = len(embedded_data[0]["vector"])
        print(f"벡터 차원(Dimension) 확인: {VECTOR_DIMENSION}")
    except Exception as e:
        print(f"오류: 첫 번째 데이터에서 벡터를 읽을 수 없습니다. {e}")
        return

    # 2. Qdrant 클라이언트 연결 및 컬렉션 설정
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        setup_qdrant_collection(client, QPOLL_COLLECTION_NAME, VECTOR_DIMENSION)
    except Exception as e:
        print(f"Qdrant 연결 실패: {e}")
        return
    
    START_BATCH_NUM = 960
    END_BATCH_NUM = 965

    START_INDEX = (START_BATCH_NUM - 1) * BATCH_SIZE
    END_INDEX = (END_BATCH_NUM) * BATCH_SIZE

    # 3. 데이터 배치(Batch) 처리 및 업로드
    # print(f"--- {BATCH_SIZE}개 단위로 Qdrant 업로드 시작 ---")
    print(f"--- {START_INDEX} 인덱스부터 이어서 업로드 시작 ---")

    data_to_upload = embedded_data[START_INDEX : END_INDEX]

    print(f"--- 총 {len(embedded_data)}개 중 {len(data_to_upload)}개 (배치 {START_BATCH_NUM}~{END_BATCH_NUM}) 업로드 시작 ---")
    
    # tqdm을 사용하여 진행률 표시
    # for i in tqdm(range(0, len(embedded_data), BATCH_SIZE), desc="Qdrant 업로드 중"):
    for i in tqdm(range(0, len(data_to_upload), BATCH_SIZE), desc="Qdrant 업로드 중"):
        
        #batch = embedded_data[i : i + BATCH_SIZE]
        batch = data_to_upload[i : i + BATCH_SIZE]
        batch_points = [] # Qdrant에 업로드할 포인트 배치

        for item in batch:
            
            # 메타데이터 (Payload) 생성 (vector와 sentence 제외)
            payload = {
                "panel_id": item.get("panel_id"),
                "question": item.get("question"),
                "sentence": item.get("sentence") # 원본 문장도 저장
            }
            
            point = models.PointStruct(
                id=str(uuid4()), # 고유 ID
                vector=item["vector"], # 저장된 벡터
                payload=payload
            )
            batch_points.append(point)

        # 4. Qdrant에 배치 업로드 (upsert)
        try:
            client.upsert(
                collection_name=QPOLL_COLLECTION_NAME,
                points=batch_points,
                wait=True
            )
        except Exception as e:
            print(f"  > 배치 {i // BATCH_SIZE + 1} 업로드 실패: {e}")
            
    print("\n--- 모든 작업 완료 ---")
    count_result = client.count(collection_name=QPOLL_COLLECTION_NAME, exact=True)
    print(f"'{QPOLL_COLLECTION_NAME}' 컬렉션에 총 {count_result.count}개의 벡터가 저장되었습니다.")

if __name__ == '__main__':
    main()
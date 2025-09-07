import streamlit as st
import boto3
import json
from typing import List, Dict
import time

# AWS Bedrock 클라이언트 초기화
@st.cache_resource
def init_bedrock_client():
    """AWS Bedrock 클라이언트 초기화"""
    try:
        client = boto3.client(
            'bedrock-runtime',
            region_name=st.secrets["AWS_DEFAULT_REGION"],  # 원하는 리전으로 변경
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        return client
    except Exception as e:
        st.error(f"AWS Bedrock 클라이언트 초기화 실패: {str(e)}")
        return None

def call_bedrock_api(client, messages: List[Dict]) -> str:
    """
    AWS Bedrock API를 호출하며, 스로틀링에 대비한 지수 백오프를 적용합니다.
    Args:
        client: The boto3 bedrock-runtime client.
        messages: 전체 대화 기록을 담은 리스트 (Streamlit session state에서 가져옴).

    Returns:
        모델의 응답 텍스트 또는 오류 메시지.
    """
    max_retries = 5
    base_delay = 1  # in seconds

    for retry_count in range(max_retries):
        try:
            # Claude 3.5 Sonnet 모델 ID 고정
            model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
            
            # Claude 모델용 요청 본문 구성
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": messages
            }
            
            # API 호출
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType='application/json'
            )
            
            # 응답 파싱
            response_body = json.loads(response['body'].read())
            
            # 디버깅: 응답 구조 확인
            # st.write("🔍 API 응답 구조:", response_body)
            
            # content 필드에서 텍스트 추출
            if 'content' in response_body and isinstance(response_body['content'], list) and len(response_body['content']) > 0:
                first_content = response_body['content'][0]
                if isinstance(first_content, dict) and 'text' in first_content:
                    return first_content['text']
            
            return "응답에서 텍스트를 찾을 수 없습니다."
            
        except client.exceptions.ThrottlingException as e:
            st.warning(f"요청이 너무 많습니다. {base_delay * (2 ** retry_count)}초 후 재시도합니다.")
            time.sleep(base_delay * (2 ** retry_count))
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}\n응답 타입: {type(e)}")
            return f"오류가 발생했습니다: {str(e)}\n응답 타입: {type(e)}"
    
    return "최대 재시도 횟수에 도달했습니다. 잠시 후 다시 시도해 주세요."

def local_chatbot_response(user_input: str) -> str:
    """로컬 테스트용 간단한 응답 함수"""
    responses = {
        "안녕": "안녕하세요! 무엇을 도와드릴까요?",
        "이름": "저는 AI 챗봇입니다.",
        "날씨": "죄송하지만 실시간 날씨 정보는 제공할 수 없습니다.",
        "시간": f"현재 시간은 대략 {time.strftime('%H:%M')} 입니다."
    }
    
    for keyword, response in responses.items():
        if keyword in user_input:
            return response
    
    return "흥미로운 질문이네요! AWS Bedrock이 연결되면 더 자세한 답변을 드릴 수 있을 것입니다."

# Streamlit 앱 구성
def main():
    st.set_page_config(
        page_title="AI 챗봇",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI 챗봇")
    st.markdown("AWS Bedrock과 연결된 질문 답변 챗봇입니다.")
    
    # 사이드바에서 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # AWS 연결 상태 확인
        use_bedrock = st.checkbox("AWS Bedrock 사용", value=False)
        
        if use_bedrock:
            st.info("🤖 사용 모델: Claude 3.5 Sonnet")
            # st.info("🔐 AWS 자격증명: secrets.toml 파일에서 로드")
        
        # 채팅 기록 초기화 버튼
        if st.button("🗑️ 채팅 기록 초기화"):
            st.session_state.messages = []
            st.rerun()
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            if use_bedrock:
                # AWS Bedrock 사용
                bedrock_client = init_bedrock_client()
                if bedrock_client:
                    with st.spinner("AI가 생각중입니다..."):
                        full_response = call_bedrock_api(bedrock_client, st.session_state.messages)
                else:
                    full_response = "AWS Bedrock 연결에 실패했습니다. 로컬 모드로 전환합니다."
                    full_response += "\n\n" + local_chatbot_response(prompt)
            else:
                # 로컬 테스트 모드
                with st.spinner("로컬 AI가 생각중입니다..."):
                    time.sleep(1)  # 실제 API 호출을 시뮬레이션
                    full_response = local_chatbot_response(prompt)
            
            # 타이핑 효과 시뮬레이션
            placeholder_text = ""
            for char in full_response:
                placeholder_text += char
                message_placeholder.markdown(placeholder_text + "▌")
                time.sleep(0.01)
            message_placeholder.markdown(full_response)
        
        # AI 응답을 세션에 추가
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

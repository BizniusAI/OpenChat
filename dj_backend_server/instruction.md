# openchat使用步驟
1. 下載專案: git@github.com:BizniusAI/OpenChat.git
2. 切換branch到develop
3. 到dj_backend_server: cd ./dj_backend_server
4. 更新 .env.docker
5. 更新 docker-compose.yml
6. build image: docker-compose build
7. 啟動docker: make install
8. 關閉docker: make down
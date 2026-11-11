package main

import (
	"log"
	"os"
	"os/exec"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// При помощи GORM создаем или проверяем что таблицы есть через AutoMigrate
// Структуру каждой таблицы можно посмотреть в файле tables.go
func InitTables() {
	err := globalDBCon.AutoMigrate(&Session{}, &Messages{}, &Leads{}, &Profiles{})
	if err != nil {
		log.Fatal("Критическая ошибка: Неудалось создать таблицы")
	}
	log.Print("Таблицы успешно созданы")
}

func InitDB() {
	log.Println("Удаление старого Образа (пока мы тестируем)")
	_ = exec.Command("docker", "compose", "down", "-v").Run()
	log.Println("Запуск docker-compose.yml")
	cmd := exec.Command("docker", "compose", "up", "-d", "db")
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("Неудалось инициализировать БД, err: %v log: %v", err, string(output))
	}
	log.Println("Ждем инициализацию БД")
	time.Sleep(5 * time.Second)
	log.Println("Контейнер успешно инициализирован")
}

// Данная функция инициализирует подключение к DB а также настраивает Connection Pool
func SetupDBConn() {
	if err := godotenv.Load(); err != nil {
		log.Fatal("Критическая ошибка: Неудалось загрузить .env")
	}
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		log.Fatal("Критическая ошибка: Неудалось найти строку подключения")
	}
	var err error
	globalDBCon, err = gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Fatalf("Критическая ошибка: неудалось подключитбся к БД %v", err)
	}
	poolDB, err := globalDBCon.DB()
	if err != nil {
		log.Fatalf("Критическая ошибка: неудалось получить объект sql.DB %v", err)
	}

	maxOpen, _ := strconv.Atoi(os.Getenv("DB_MAX_OPEN_CONNS"))
	if maxOpen == 0 {
		maxOpen = 30
	}
	poolDB.SetMaxOpenConns(maxOpen)

	maxIdle, _ := strconv.Atoi(os.Getenv("DB_MAX_IDLE_CONNS"))
	if maxIdle == 0 {
		maxIdle = 10
	}
	poolDB.SetMaxIdleConns(maxIdle)

	poolDB.SetConnMaxLifetime(30 * time.Minute)

	poolDB.SetConnMaxIdleTime(5 * time.Minute)
	log.Print("Подключение успешно настроено")
}

// Данная функция находит все записи из таблицы с полями которые указываются типом T где T -- interface{}
func GetAllFromDB[T any](session_id string, table interface{}) ([]T, error) {
	var result []T
	err := globalDBCon.Model(table).
		Where("session_id = ?", session_id).
		Find(&result).Error
	return result, err
}

// Данная функция находит одну запись из таблицы с полями которые указываются типом T где T -- interface{}
func GetOneFromDB[T any](session_id string, table interface{}) (T, error) {
	var result T
	err := globalDBCon.Model(table).
		Where("session_id = ?", session_id).
		Find(&result).Error
	return result, err
}

// Данная функция необходима для создания Корректного запроса к
// /v1/chat и возвращения ответа в handler (Наверное готово)
func GetAIAnwser(json_data UserRequest) (AIResponseForm, error) {
	var ai_request RequestToAI
	ai_request.SessionId = json_data.SessionId
	ai_request.RequestId = json_data.RequestId
	ai_request.Message = json_data.Message
	history, err := GetAllFromDB[AiHistoryMessageForm](json_data.SessionId, &Messages{})
	if err != nil {
		return AIResponseForm{}, err
	}
	ai_request.History = history

	customer_profile, err := GetOneFromDB[CustomerProfileForm](json_data.SessionId, &Profiles{})
	if err != nil {
		return AIResponseForm{}, err
	}
	ai_request.CustomerProfile = customer_profile
	var ai_response AIResponseForm
	_, err = globalAIClient.R().
		SetBody(ai_request).
		SetResult(&ai_response).
		Post("http://localhost/v1/chat")
	if err != nil {
		return AIResponseForm{}, err
	}
	return ai_response, nil
}

// Данная функция вставляет данные в Таблицу
// Можно запенить на gorm.Model().Save()
func InsertUserMessage(json_data UserRequest) error {
	var user_message Messages
	user_message.MessageId = uuid.NewString()
	user_message.SessionId = json_data.SessionId
	user_message.Role = "user"
	user_message.Action = ""
	user_message.Content = json_data.SessionId
	user_message.CreatedAt = time.Now()
	err := globalDBCon.Create(&user_message).Error
	return err
}

package main

import (
	"log"
	"os"
	"os/exec"
	"strconv"
	"time"
	"errors"
	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"github.com/davecgh/go-spew/spew"
	"github.com/gin-gonic/gin"
)

// При помощи GORM создаем или проверяем что таблицы есть через AutoMigrate
// Структуру каждой таблицы можно посмотреть в файле tables.go
func InitTables() {
	err := globalDBCon.AutoMigrate(&Customers{}, &Cookies{}, &Session{}, &Messages{}, &Leads{}, &Profiles{})
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
func GetAllFromDB[T any](where_query string, where_value string, table interface{}) ([]T, error) {
	var result []T
	err := globalDBCon.Model(table).
		Where(where_query, where_value).
		Find(&result).Error
	return result, err
}
//Where("session_id = ?", session_id)
// Данная функция находит одну запись из таблицы с полями которые указываются типом T где T -- interface{}
func GetOneFromDB[T any](where_query string, where_value string, table interface{}) (T, error) {
	var result T
	err := globalDBCon.Model(table).
		Where(where_query, where_value).
		Find(&result).Error
	return result, err
}

func GetCustomerIdFromDB(ctx *gin.Context) (string, error){
	val, exists := ctx.Get(CookieName)
	if !exists {
		err := errors.New("Cookie_id wasn't exist in context map")
		log.Printf("Неудалось найти cookie_id в хранилище контекста | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		return "", err
	}
	cookie_id, ok := val.(string)

	if !ok {
		err := errors.New("Can't cast cookie_id value to string")
		log.Printf("Неудалось правильно преобразовать cookie_id из контекста | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		return "", err
	}

	cookie_struct, err := GetOneFromDB[Cookies]("cookie_id = ?", cookie_id, &Cookies{})
	if err != nil {
		log.Printf("Неудалось найти Cookie в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.SetCookie(CookieName, "", -1, "/", "", false, true)
		return "" , err
	}
	return  cookie_struct.CustomerId, nil

}

// Данная функция необходима для создания Корректного запроса к
// /v1/chat и возвращения ответа в handler (Наверное готово)
func GetAIAnwser(json_data UserRequest) (AIResponseForm, error) {
	var ai_request RequestToAI
	ai_request.SessionId = json_data.SessionId
	ai_request.RequestId = json_data.RequestId
	ai_request.Message = json_data.Message
	history, err := GetAllFromDB[AiHistoryMessageForm]("session_id = ?", json_data.SessionId, &Messages{})
	if err != nil {
		log.Printf("Ошибка при поиске истории сессии | error: %s", err.Error())
		return AIResponseForm{}, err
	}
	ai_request.History = history

	customer_profile, err := GetOneFromDB[CustomerProfileForm]("session_id = ?", json_data.SessionId, &Profiles{})
	if err != nil {
		log.Printf("Ошибка при поиске профиля | error: %s", err.Error())
		return AIResponseForm{}, err
	}
	ai_request.CustomerProfile = customer_profile
	var ai_response AIResponseForm
	_, err = globalAIClient.R().
		SetBody(ai_request).
		SetResult(&ai_response).
		Post("http://localhost/v1/chat")
	if err != nil {
		log.Printf("Ошибка при отправке запроса ИИ | error: %s", err.Error())
		return AIResponseForm{}, err
	}
	return ai_response, nil
}

// Данная функция вставляет данные в Таблицу
func InsertMessage(json_data any) error {
	message := Messages {
		MessageId: uuid.NewString(),
		CreatedAt: time.Now(),
	}
	switch type_struct := json_data.(type) {
	case UserRequest :
		message.SessionId = type_struct.SessionId
		message.Role = "user"
		message.Action = ""
		message.Content = type_struct.Message
	case AIResponseForm :
		message.SessionId = type_struct.SessionId
		message.Role = "ai"
		message.Action = type_struct.Action
		message.Content = type_struct.Answer
	}
	err := globalDBCon.Create(&message).Error
	return err
}


func initLogger(){
	spew.Config.Indent = "  "
	spew.Config.DisablePointerAddresses = true
}

func LogStruct(label string, data interface{}){
	log.Printf(" %s\n%s", label, spew.Sdump(data))
}


func FillErrorForm(title string, status int, err error) (interface{}){
	var error_struct ErrorForm
	error_struct.Type = "about:blank"
	error_struct.Title = title
	error_struct.Status = status
	error_struct.Detail = err.Error()
    return  error_struct
}
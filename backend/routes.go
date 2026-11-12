package main

import (
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

var CookieName = "guest_id"
//Создаем Cookie при первом обращении 
func CreateCookieMiddleWare() gin.HandlerFunc {
	return func(ctx *gin.Context){
		//Пытаемся найти Cookie если ее нет то создаем на неделю
		id, err := ctx.Cookie(CookieName)
		if (err != nil || id == "") {
			id = uuid.NewString();
			ctx.SetCookie(CookieName, id, 3600 * 24 * 7, "/", "", false, false)
		}
		//Записать в БД Cookie
		

		//Записываем в словарик контекса
		ctx.Set("guest_id", id)
		ctx.Next()
	}
}
//Проверяем есть ли Cookie для необходимых эндпоинтов
func ReadCookieMiddleWare() gin.HandlerFunc {
	return func(ctx *gin.Context){
		id, err := ctx.Cookie(CookieName)
		if (err != nil || id == ""){
			ctx.JSON(401, FillErrorForm("User Anauthorized (No Cookie)", 401, err))
			return 
		} 
		ctx.Set("guest_id", id)
		ctx.Next()
	}
}


func SetupRoutes() *gin.Engine {
	// Для работы с БД будет исользоваться GORM из-за удобства автоматически создавать таблицы и записывать туда данные создавая их из type struct
	gin.SetMode(gin.DebugMode)
	r := gin.Default()
	// Политика CORS чтобы доверять грубо говоря ip, пока что localhost можно поменять позже, пока не понятно где находится front в будущем
	config := cors.Config {
		AllowOrigins: []string{"http://localhost:3000"},
		AllowMethods: []string{"POST", "GET", "OPTIONS"},
		AllowHeaders: []string{"Origin", "Content-type", "Authorization"},
		AllowCredentials: true,
		MaxAge: 12 * time.Hour,
	}
	r.Use(cors.New(config))


	api := r.Group("/api")
	{
		// Публичные маршруты
		api.GET("/hello", Hello)

		//Маршрут с созданием Cookie
		api.POST("/sessions", CreateCookieMiddleWare(), CreateSessionHandler)

		// Группа с проверка на наличие Cookie
		cookieGroup := api.Group("", ReadCookieMiddleWare())
		{
			cookieGroup.GET("/sessions/:session_id/messages", ChatHistoryHandler)
			cookieGroup.POST("/chat", UserMessageHandler)
			cookieGroup.POST("/leads", ManagerStatmentHandler)
		}
	}
	return r
}
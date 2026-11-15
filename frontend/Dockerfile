# ЭТАП 1: Сборка приложения
FROM node:20-alpine AS builder

WORKDIR /app

# Копируем файлы зависимостей
COPY package.json package-lock.json ./

# Устанавливаем зависимости (используем ci для точного соответствия версий)
RUN npm ci

# Копируем весь остальной код
COPY . .

# Принимаем переменную окружения для API (важно для Vite!)
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# Собираем проект (получаем папку dist)
RUN npm run build

# ЭТАП 2: Раздача статики через Nginx
FROM nginx:alpine

# Копируем собранные файлы из первого этапа в папку Nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Копируем наш кастомный конфиг Nginx (создадим ниже)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
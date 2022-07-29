import telebot
from telebot import types
import config
from styletransfer import Run_ST_Process
from upscale import result_photo

bot = telebot.TeleBot(config.TOKEN)

in_work = False


def save_photo(message):
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = "tmp/" + message.document.file_name
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    return src


@bot.message_handler()
def send_info(message):
    if in_work:
        msg = "Работаю..."
    else:
        msg = "Пришли фото для переноса стиля"
    bot.send_message(message.chat.id, msg)


@bot.message_handler(content_types=['document'])
def handle_base_photo(message):
    global in_work
    try:
        src_base = save_photo(message)
        rmk = types.ReplyKeyboardMarkup(resize_keyboard=True)
        rmk.add(types.KeyboardButton("Отмена"))
        msg = bot.send_message(message.chat.id, "Пришли стиль", reply_markup=rmk)
        bot.register_next_step_handler(msg, handle_style_photo, src_base)
    except Exception as e:
        bot.reply_to(message, e)


def handle_style_photo(message, src_base):
    global in_work
    remove_k = types.ReplyKeyboardRemove()
    if message.text == "Отмена":
        bot.send_message(message.chat.id, "Операция отменена", reply_markup=remove_k)
    else:
        src_style = save_photo(message)
        in_work = True
        bot.send_message(message.chat.id, "Обрабатываю...", reply_markup=remove_k)
        if Run_ST_Process(src_base, src_style):
            result_photo("tmp/result.jpg", 2)
            bot.send_photo(message.chat.id, open("tmp/result.jpg", 'rb'))
        else:
            bot.send_message(message.chat.id, "Ошибка", reply_markup=remove_k)
        in_work = False


bot.polling()

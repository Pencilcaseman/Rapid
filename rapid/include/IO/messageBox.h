#pragma once

#include "../internal.h"

namespace rapid
{
	namespace message
	{
	#ifdef RAPID_DEBUG
	#define rapidAssert(cond, err) { if (!(cond)) {rapid::message::RapidError("Assertion Failed", err).display(); }}

	#else
	#define rapidAssert(cond, err)
	#endif

		inline void rapidValidate(bool condition, const std::string &err = "Error", const int code = 1)
		{
			if (!condition)
			{
				std::cerr << err << "\n";
				exit(code);
			}
		}
	}
}

#ifdef RAPID_OS_WINDOWS
#include <WinUser.h>

namespace rapid
{
	namespace message
	{
		class RapidMessageBox
		{
		public:
			enum class MessageBoxType : int
			{
				ICON_ERROR = MB_ICONERROR,
				ICON_QUESTION = MB_ICONQUESTION,
				ICON_WARNING = MB_ICONWARNING,
				ICON_INFORMATION = MB_ICONINFORMATION,

				BUTTON_ABORD_RETRY_IGNORE = MB_ABORTRETRYIGNORE,
				BUTTON_CANCEL_TRY_CONTINUE = MB_CANCELTRYCONTINUE,
				BUTTON_HELP = MB_HELP,
				BUTTON_OK = MB_OK,
				BUTTON_OK_CANCEL = MB_OKCANCEL,
				BUTTON_RETRY_CANCEL = MB_RETRYCANCEL,
				BUTTON_YES_NO = MB_YESNO,
				BUTTON_YES_NO_CANCEL = MB_YESNOCANCEL,

				DEFAULT_FIRST = MB_DEFBUTTON1,
				DEFAULT_SECOND = MB_DEFBUTTON2,
				DEFAULT_THIRD = MB_DEFBUTTON3,

				RETURN_ABORT = IDABORT,
				RETURN_CANCEL = IDCANCEL,
				RETURN_CONTINUE = IDCONTINUE,
				RETURN_IGNORE = IDIGNORE,
				RETURN_NO = IDNO,
				RETURN_OK = IDOK,
				RETURN_RETRY = IDRETRY,
				RETURN_TRY_AGAIN = IDTRYAGAIN,
				RETURN_YES = IDYES
			};

			std::string title = "Rapid Message Box";
			std::string message = "Message Box";
			MessageBoxType icon = MessageBoxType::ICON_INFORMATION;
			MessageBoxType buttons = MessageBoxType::BUTTON_YES_NO_CANCEL;
			MessageBoxType defaultButton = MessageBoxType::DEFAULT_FIRST;

			RapidMessageBox() = default;

			RapidMessageBox(const std::string &t_,
							const std::string &message_ = "Message Box",
							const MessageBoxType icon_ = MessageBoxType::ICON_INFORMATION,
							const MessageBoxType buttons_ = MessageBoxType::BUTTON_YES_NO_CANCEL,
							const MessageBoxType defaultButton_ = MessageBoxType::DEFAULT_FIRST)
			{
				title = t_;
				message = message_;
				icon = icon_;
				buttons = buttons_;
				defaultButton = defaultButton_;
			}

			inline virtual bool pressAbort()
			{
				return true;
			}

			inline virtual bool pressCancel()
			{
				return true;
			}

			inline virtual bool pressContinue()
			{
				return true;
			}

			inline virtual bool pressIgnore()
			{
				return true;
			}

			inline virtual bool pressNo()
			{
				return true;
			}

			inline virtual bool pressOk()
			{
				return true;
			}

			inline virtual bool pressRetry()
			{
				return true;
			}

			inline virtual bool pressTryAgain()
			{
				return true;
			}

			inline virtual bool pressYes()
			{
				return true;
			}

			inline virtual bool error()
			{
				return true;
			}

			MessageBoxType display()
			{
				const auto m = (LPCSTR) message.c_str();
				const auto t = (LPCSTR) title.c_str();

				int msgBoxID = MessageBox(
					nullptr,
					m,
					t,
					(int) ((int) icon | (int) buttons | (int) defaultButton)
				);

				bool errorOccured = false;

				switch (msgBoxID)
				{
					case (int) MessageBoxType::RETURN_ABORT:
						if (!pressAbort())
							errorOccured = true;
						break;
					case (int) MessageBoxType::RETURN_CANCEL:
						if (!pressCancel())
							errorOccured = true;
						break;
					case (int) MessageBoxType::RETURN_CONTINUE:
						if (!pressContinue())
							errorOccured = true;
						break;
					case (int) MessageBoxType::RETURN_IGNORE:
						if (!pressIgnore())
							errorOccured = true;
						break;
					case (int) MessageBoxType::RETURN_NO:
						if (!pressNo())
							errorOccured = true;
						break;
					case (int) MessageBoxType::RETURN_OK:
						if (!pressOk())
							errorOccured = true;
						break;
					case (int) MessageBoxType::RETURN_RETRY:
						if (!pressRetry())
							errorOccured = true;
						break;
					case (int) MessageBoxType::RETURN_TRY_AGAIN:
						if (!pressTryAgain())
							errorOccured = true;
						break;
					case (int) MessageBoxType::RETURN_YES:
						if (!pressYes())
							errorOccured = true;
						break;
				}

				if (errorOccured)
				{
					rapidValidate(error(), "Message box failed");
				}

				return (MessageBoxType) msgBoxID;
			}
		};

		class RapidError : public RapidMessageBox
		{
		public:
			RapidError(const std::string &errorType_,
					   const std::string &errorMessage_)
			{
				title = errorType_;
				message = errorMessage_;

				buttons = RapidMessageBox::MessageBoxType::BUTTON_OK;
				icon = RapidMessageBox::MessageBoxType::ICON_ERROR;
			}

			bool pressOk() override
			{
				std::cerr << "Something went wrong\n";
				exit(1);

				return true;
			}
		};

		class RapidWarning : public RapidMessageBox
		{
		public:
			RapidWarning(const std::string &errorType_,
						 const std::string &errorMessage_,
						 const std::string &question = "Would you like to exit?")
			{
				title = errorType_;
				message = errorMessage_ + "\n\n" + question;

				buttons = RapidMessageBox::MessageBoxType::BUTTON_YES_NO;
				icon = RapidMessageBox::MessageBoxType::ICON_WARNING;
			}

			bool pressYes() override
			{
				std::cerr << "Warning failed\n";
				exit(1);

				return true;
			}

			bool pressNo() override
			{
				return true;
			}
		};
	}
}
#else
#include "../rapid_math.h"

namespace rapid
{
	namespace message
	{
		class RapidMessageBox
		{
		public:
			enum class MessageBoxType : int
			{
				ICON_ERROR					= 0,
				ICON_QUESTION				= 1,
				ICON_WARNING				= 2,
				ICON_INFORMATION			= 3,
											  
				BUTTON_ABORD_RETRY_IGNORE	= 4,
				BUTTON_CANCEL_TRY_CONTINUE	= 5,
				BUTTON_HELP					= 6,
				BUTTON_OK					= 7,
				BUTTON_OK_CANCEL			= 8,
				BUTTON_RETRY_CANCEL			= 9,
				BUTTON_YES_NO				= 10,
				BUTTON_YES_NO_CANCEL		= 11,
											  
				DEFAULT_FIRST				= 12,
				DEFAULT_SECOND				= 13,
				DEFAULT_THIRD				= 14,
											  
				RETURN_ABORT				= 15,
				RETURN_CANCEL				= 16,
				RETURN_CONTINUE				= 17,
				RETURN_IGNORE				= 18,
				RETURN_NO					= 19,
				RETURN_OK					= 20,
				RETURN_RETRY				= 21,
				RETURN_TRY_AGAIN			= 22,
				RETURN_YES					= 23
			};

			std::string title = "Rapid Message Box";
			std::string message = "Message Box";
			MessageBoxType icon = MessageBoxType::ICON_INFORMATION;
			MessageBoxType buttons = MessageBoxType::BUTTON_YES_NO_CANCEL;
			MessageBoxType defaultButton = MessageBoxType::DEFAULT_FIRST;

			RapidMessageBox() = default;

			RapidMessageBox(const std::string &t_,
							const std::string &message_ = "Message Box",
							const MessageBoxType icon_ = MessageBoxType::ICON_INFORMATION,
							const MessageBoxType buttons_ = MessageBoxType::BUTTON_YES_NO_CANCEL,
							const MessageBoxType defaultButton_ = MessageBoxType::DEFAULT_FIRST)
			{
				title = t_;
				message = message_;
				icon = icon_;
				buttons = buttons_;
				defaultButton = defaultButton_;
			}

			inline virtual bool pressAbort()
			{
				return true;
			}

			inline virtual bool pressCancel()
			{
				return true;
			}

			inline virtual bool pressContinue()
			{
				return true;
			}

			inline virtual bool pressIgnore()
			{
				return true;
			}

			inline virtual bool pressNo()
			{
				return true;
			}

			inline virtual bool pressOk()
			{
				return true;
			}

			inline virtual bool pressRetry()
			{
				return true;
			}

			inline virtual bool pressTryAgain()
			{
				return true;
			}

			inline virtual bool pressYes()
			{
				return true;
			}

			inline virtual bool error()
			{
				return true;
			}

			MessageBoxType display()
			{
				// Find the longest value
				auto titleLen = title.length();
				auto msgLen = message.length();

				auto maxLen = rapid::math::min(rapid::math::max(titleLen, msgLen), 50);

				std::cout << "\n\n" << std::string(maxLen + 4, '=') << "\n";

				auto pad = std::string((maxLen - titleLen + 2) / 2, ' ');
				std::cout << "# " << pad << title << std::string((maxLen - titleLen) / 2, ' ') << "#\n";

				std::cout << std::string(maxLen + 4, '-') << "\n";

				if (msgLen <= maxLen)
				{
					std::cout << "# " << message << " #\n";
				}
				else
				{
					// Split into segments that are at most 50 characters wide
					std::istringstream iss(message);
					std::vector<std::string> words((std::istream_iterator<std::string>(iss)),
												   std::istream_iterator<std::string>());

					std::vector<std::string> segments;

					for (const auto &word : words)
					{
						if (segments.empty())
							segments.emplace_back(word + " ");
						else if (segments[segments.size() - 1].empty())
							segments[segments.size() - 1] += word + " ";
						else
						{
							if ((segments[segments.size() - 1] + word).length() < 50)
								segments[segments.size() - 1] += word + " ";
							else
								segments.emplace_back(word + " ");
						}
					}

					for (const auto &seg : segments)
					{
						auto padf = std::string((maxLen - seg.length() + 2) / 2, ' ');
						auto padb = std::string(maxLen - padf.length() - (seg.length() - 2), ' ');
						std::cout << "#" << padf << seg << padb << "#\n";
					}
				}

				std::cout << std::string(maxLen + 4, '=') << "\n";

				std::cout << "Press enter to accept";
				auto res = std::getchar();

				pressOk();

				return MessageBoxType::BUTTON_OK;
			}
		};

		class RapidError : public RapidMessageBox
		{
		public:
			RapidError(const std::string &errorType_,
					   const std::string &errorMessage_)
			{
				title = errorType_;
				message = errorMessage_;

				buttons = RapidMessageBox::MessageBoxType::BUTTON_OK;
				icon = RapidMessageBox::MessageBoxType::ICON_ERROR;
			}

			bool pressOk() override
			{
				std::cerr << title << " : FAILED\n";
				exit(1);

				return true;
			}
		};

		class RapidWarning : public RapidMessageBox
		{
		public:
			RapidWarning(const std::string &errorType_,
						 const std::string &errorMessage_,
						 const std::string &question = "Would you like to exit?")
			{
				title = errorType_;
				message = errorMessage_ + "\n\n" + question;

				buttons = RapidMessageBox::MessageBoxType::BUTTON_YES_NO;
				icon = RapidMessageBox::MessageBoxType::ICON_WARNING;
			}

			bool pressYes() override
			{
				std::cerr << title << " : FAILED\n";
				exit(1);

				return true;
			}

			bool pressNo() override
			{
				return true;
			}
		};
	}
}
#endif

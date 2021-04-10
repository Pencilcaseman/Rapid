#pragma once

#include "../internal.h"
#include "utils.h"

namespace rapid
{
	namespace parser
	{
		class ExpressionSolver
		{
		public:
			std::string expression;
			std::vector<std::string> infix;
			std::vector<std::string> postfix;
			std::vector<std::pair<float64, std::string>> processed;

			// std::vector<std::string> splitBy = {" ", "(", ")", "+", "-", "*", "/", "^", "%"};
			std::vector<std::string> splitBy = {" ", "(", ")", ">", "<", "=", "!", "+", "-", "*", "/", "^", "%"};
			std::unordered_map<std::string, float64> variables;

			// Basic math operations
			std::vector<std::string> operators = {"^", "*", "/", "%", "-", "+"};

			// Functions and their definitions
			std::vector<std::string> functionNames;
			std::vector<std::function<float64(float64)>> functionDefinitions;

			// Error tracking
			bool errorOccured = false;

		public:

			ExpressionSolver() = default;

			ExpressionSolver(const std::string &expr) : expression(expr)
			{
				operators.insert(operators.begin(), functionNames.begin(), functionNames.end());

				// Register common functions
				registerFunction("sin", [](float64 x)
				{
					return std::sin(x);
				});

				registerFunction("cos", [](float64 x)
				{
					return std::cos(x);
				});

				registerFunction("tan", [](float64 x)
				{
					return std::tan(x);
				});

				registerFunction("asin", [](float64 x)
				{
					return std::asin(x);
				});

				registerFunction("acos", [](float64 x)
				{
					return std::acos(x);
				});

				registerFunction("atan", [](float64 x)
				{
					return std::atan(x);
				});

				registerFunction("exp", [](float64 x)
				{
					return std::exp(x);
				});
			}

			inline void expressionToInfix()
			{
				uint64 i = 0;
				bool append = false;
				for (const auto &term : splitString(expression, splitBy))
				{
					if (term != " " && !term.empty())
					{
						if (!append)
							infix.emplace_back(term);
						else
						{
							infix[i - 1] += term;
							append = false;
						}

						if (i == 0 && (term == "+" || term == "-"))
							append = true;
						else if (i > 1 && (term == "+" || term == "-") && (std::find(operators.end() - 6, operators.end(), infix[infix.size() - 2]) != operators.end()))
							append = true;

						i++;
					}
				}
			}

			inline void processInfix()
			{
				std::vector<std::string> newInfix;

				for (uint64 i = 0; i < infix.size(); i++)
				{
					bool modified = false;

					if (infix[i] == ">")
					{
						if (i < infix.size() - 1 && infix[i + 1] == "=")
						{
							newInfix.emplace_back(">=");
							modified = true;
							i++;
						}
						else
						{
							newInfix.emplace_back(">");
							modified = true;
						}
					}

					if (infix[i] == "<")
					{
						if (i < infix.size() - 1 && infix[i + 1] == "=")
						{
							newInfix.emplace_back("<=");
							modified = true;
							i++;
						}
						else
						{
							newInfix.emplace_back("<");
							modified = true;
						}
					}

					if (i < infix.size() - 1 && infix[i] == "!" && infix[i + 1] == "=")
					{
						newInfix.emplace_back(">=");
						modified = true;
						i++;
					}

					if (!modified)
						newInfix.emplace_back(infix[i]);
				}

				infix = newInfix;
			}

			inline void infixToPostfix()
			{
				std::stack<std::string> stack;

				for (const auto &token : infix)
				{
					auto it = std::find(functionNames.begin(), functionNames.end(), token);

					if (it == functionNames.end() && isalphanum(token))
						postfix.emplace_back(token);
					else if (token == "(" || token == "^")
						stack.push(token);
					else if (token == ")")
					{
						while (!stack.empty() && stack.top() != "(")
						{
							postfix.emplace_back(stack.top());
							stack.pop();
						}
						stack.pop();
					}
					else
					{
						while (!stack.empty() && std::find(operators.begin(), operators.end(), token) >= std::find(operators.begin(), operators.end(), stack.top()))
						{
							postfix.emplace_back(stack.top());
							stack.pop();
						}
						stack.push(token);
					}
				}

				while (!stack.empty())
				{
					postfix.emplace_back(stack.top());
					stack.pop();
				}
			}

			inline void postfixProcess()
			{
				for (const auto &term : postfix)
				{
					if (isnum(term))
						processed.emplace_back(std::make_pair(std::stod(term), ""));
					else
						processed.emplace_back(std::make_pair(0., term));
				}
			}

			inline float64 postfixEval()
			{
				std::stack<float64> stack;

				for (const auto &term : processed)
				{
					bool evaluated = false;

					if (term.second.length() == 0)
					{
						stack.push(term.first);
						evaluated = true;
					}

					if (!evaluated)
					{
						std::string varname;
						float64 mult = 1;

						if (term.second[0] == '-')
						{
							varname = std::string(term.second.begin() + 1, term.second.end());
							mult = -1;
						}
						else if (term.second[0] == '+')
						{
							varname = std::string(term.second.begin() + 1, term.second.end());
						}
						else
						{
							varname = term.second;
						}

						if (variables.find(varname) != variables.end())
						{
							stack.push(variables.at(varname) * mult);
							evaluated = true;
						}
						else
						{
							errorOccured = true;
						}
					}

					if (!evaluated)
					{
						float64 a = 0;
						float64 b = stack.top(); stack.pop();

						// Function
						for (uint64 i = 0; i < functionDefinitions.size(); i++)
						{
							if (term.second == functionNames[i])
							{
								stack.push(functionDefinitions[i](b));
								evaluated = true;
								break;
							}
						}

						if (!stack.empty() && !evaluated)
						{
							a = stack.top(); stack.pop();

							if (term.second == "+")
							{
								stack.push(a + b);
								evaluated = true;
							}
							else if (term.second == "-")
							{
								stack.push(a - b);
								evaluated = true;
							}
							else if (term.second == "*")
							{
								stack.push(a * b);
								evaluated = true;
							}
							else if (term.second == "/")
							{
								stack.push(a / b);
								evaluated = true;
							}
							else if (term.second == "^")
							{
								stack.push(std::pow(a, b));
								evaluated = true;
							}
							else if (term.second == "%")
							{
								stack.push(std::fmod(a, b));
								evaluated = true;
							}
							else if (term.second == ">")
							{
								stack.push(a > b);
								evaluated = true;
							}
							else if (term.second == "<")
							{
								stack.push(a < b);
								evaluated = true;
							}
							else if (term.second == ">=")
							{
								stack.push(a >= b);
								evaluated = true;
							}
							else if (term.second == "<=")
							{
								stack.push(a <= b);
								evaluated = true;
							}
							else if (term.second == "!=")
							{
								stack.push(a != b);
								evaluated = true;
							}
							else
							{
								errorOccured = true;
							}
						}
					}
				}

				return stack.top();
			}

			template<typename lambda>
			inline void registerFunction(const std::string &name, const lambda &func)
			{
				functionNames.emplace_back(name);
				functionDefinitions.emplace_back(std::function<float64(float64)>(func));

				operators.insert(operators.begin(), name);
			}

			inline void compile()
			{
				expressionToInfix();
				processInfix();
				infixToPostfix();
				postfixProcess();
			}

			inline float64 eval()
			{
				return postfixEval();
			}

			operator bool() const
			{
				return errorOccured;
			}
		};
	}
}
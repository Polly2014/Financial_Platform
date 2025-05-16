"""
LLM-based financial report analysis service.
使用大语言模型进行财报智能分析
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
import litellm
from litellm import completion

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LLMAnalysisService")

class LLMAnalysisService:
    """使用大语言模型进行财报智能分析的服务类"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        初始化LLM分析服务
        
        Args:
            model_name: 使用的模型名称，默认为gpt-3.5-turbo
            api_key: API密钥，如果为None则尝试从环境变量获取
        """
        self.model_name = model_name
        
        # 设置API密钥
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # 配置litellm
        litellm.drop_params = True  # 删除不支持的参数
        litellm.set_verbose = False  # 关闭详细日志
        
        logger.info(f"LLM分析服务初始化完成，使用模型: {model_name}")
    
    def analyze_financial_report(self, report_content: str, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM分析财务报告内容
        
        Args:
            report_content: 财报文本内容
            company_info: 公司基本信息，包含公司名称、股票代码、报告期间等
            
        Returns:
            包含分析结果的字典
        """
        try:
            # 准备提示词
            prompt = self._prepare_analysis_prompt(report_content, company_info)
            
            # 调用LLM进行分析
            logger.info(f"开始分析 {company_info.get('company_name', '')} 的财报")
            
            # 检查是否有有效的API密钥
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key != "sk-demo-key-please-replace-in-production":
                # 使用litellm调用API
                response = completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                analysis_result = self._parse_llm_response(response)
            else:
                # 使用模拟数据
                analysis_result = self._get_mock_analysis_result(company_info)
            
            # 添加元数据
            analysis_result["metadata"] = {
                "model": self.model_name,
                "company_name": company_info.get("company_name", ""),
                "stock_code": company_info.get("stock_code", ""),
                "report_period": company_info.get("report_period", ""),
                "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"财报分析完成: {company_info.get('company_name', '')}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"LLM分析过程中出错: {str(e)}")
            return {
                "status": "error",
                "message": f"分析失败: {str(e)}",
                "metadata": {
                    "model": self.model_name,
                    "company_name": company_info.get("company_name", ""),
                    "stock_code": company_info.get("stock_code", ""),
                    "report_period": company_info.get("report_period", "")
                }
            }
    
    def generate_investment_advice(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于财报分析结果生成投资建议
        
        Args:
            analysis_result: 财报分析结果
            
        Returns:
            包含投资建议的字典
        """
        try:
            # 准备提示词
            prompt = self._prepare_investment_prompt(analysis_result)
            
            # 调用LLM生成投资建议
            company_name = analysis_result.get("metadata", {}).get("company_name", "")
            logger.info(f"开始为 {company_name} 生成投资建议")
            
            # 检查是否有有效的API密钥
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key != "sk-demo-key-please-replace-in-production":
                # 使用litellm调用API
                response = completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_investment_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                investment_advice = self._parse_investment_response(response)
            else:
                # 使用模拟数据
                investment_advice = self._get_mock_investment_advice(analysis_result)
            
            # 添加元数据
            investment_advice["metadata"] = {
                "model": self.model_name,
                "company_name": company_name,
                "advice_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"投资建议生成完成: {company_name}")
            return investment_advice
            
        except Exception as e:
            logger.error(f"生成投资建议过程中出错: {str(e)}")
            return {
                "status": "error",
                "message": f"生成投资建议失败: {str(e)}"
            }
    
    def compare_reports(self, current_report: Dict[str, Any], previous_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较两期财报，分析变化趋势
        
        Args:
            current_report: 当前财报分析结果
            previous_report: 上一期财报分析结果
            
        Returns:
            包含比较分析的字典
        """
        try:
            # 准备提示词
            prompt = self._prepare_comparison_prompt(current_report, previous_report)
            
            # 调用LLM进行比较分析
            company_name = current_report.get("metadata", {}).get("company_name", "")
            logger.info(f"开始比较 {company_name} 的财报")
            
            # 检查是否有有效的API密钥
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key != "sk-demo-key-please-replace-in-production":
                # 使用litellm调用API
                response = completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_comparison_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=3000
                )
                comparison_result = self._parse_comparison_response(response)
            else:
                # 使用模拟数据
                comparison_result = self._get_mock_comparison_result(current_report, previous_report)
            
            # 添加元数据
            comparison_result["metadata"] = {
                "model": self.model_name,
                "company_name": company_name,
                "current_period": current_report.get("metadata", {}).get("report_period", ""),
                "previous_period": previous_report.get("metadata", {}).get("report_period", ""),
                "comparison_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"财报比较分析完成: {company_name}")
            return comparison_result
            
        except Exception as e:
            logger.error(f"比较财报过程中出错: {str(e)}")
            return {
                "status": "error",
                "message": f"比较分析失败: {str(e)}"
            }
            
    def _get_mock_analysis_result(self, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟的分析结果（用于演示）"""
        company_name = company_info.get("company_name", "示例公司")
        stock_code = company_info.get("stock_code", "000000")
        report_period = company_info.get("report_period", "2023年年度报告")
        
        return {
            "status": "success",
            "summary": f"{company_name}（{stock_code}）在{report_period}中展现了稳健的财务表现。营业收入同比增长15.2%，达到52.3亿元；净利润同比增长12.8%，达到8.7亿元。公司毛利率保持在32.5%的水平，净资产收益率为18.2%。",
            "financial_indicators": {
                "revenue": {
                    "value": "52.3亿元",
                    "change": "+15.2%",
                    "trend": "up"
                },
                "net_profit": {
                    "value": "8.7亿元",
                    "change": "+12.8%",
                    "trend": "up"
                },
                "eps": {
                    "value": "1.23元",
                    "change": "+10.5%",
                    "trend": "up"
                },
                "roe": {
                    "value": "18.2%",
                    "change": "+0.5%",
                    "trend": "up"
                },
                "gross_margin": {
                    "value": "32.5%",
                    "change": "-0.8%",
                    "trend": "down"
                },
                "debt_ratio": {
                    "value": "45.3%",
                    "change": "+2.1%",
                    "trend": "up"
                }
            },
            "business_analysis": {
                "main_business": f"{company_name}主要从事高科技制造业，核心产品包括智能设备和工业自动化解决方案。",
                "revenue_structure": {
                    "智能设备": "65%",
                    "工业自动化": "25%",
                    "技术服务": "10%"
                },
                "market_position": "在国内市场占有率约为18%，位居行业第二位。",
                "competitive_advantage": "技术创新能力强，拥有多项核心专利，产品质量稳定可靠。",
                "challenges": "面临国际竞争加剧和原材料成本上升的挑战。"
            },
            "risk_factors": [
                "原材料价格波动风险",
                "技术迭代风险",
                "市场竞争加剧风险",
                "汇率波动风险",
                "政策法规变动风险"
            ],
            "future_outlook": {
                "short_term": "预计未来一年收入增长将保持在12%-15%区间，毛利率可能小幅下降。",
                "medium_term": "公司计划扩大海外市场份额，预计三年内海外收入占比将从目前的20%提升至30%。",
                "long_term": "将持续加大研发投入，向高端制造领域拓展，培育新的增长点。"
            }
        }
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一位专业的财务分析师，擅长分析上市公司财务报告并提取关键财务指标和业务洞察。
请根据提供的财报内容，进行全面分析并以JSON格式输出结果。你的分析应当客观、准确、专业，避免主观臈断。
分析应包括但不限于：基本财务指标、盈利能力、偿债能力、运营效率、现金流状况、业务亮点与风险、未来展望等。
请确保输出格式为有效的JSON，并包含所有关键财务数据和分析结果。"""

    def _get_investment_system_prompt(self) -> str:
        """获取投资建议系统提示词"""
        return """你是一位资深投资顾问，擅长基于财务分析结果提供投资建议。
请根据提供的财报分析结果，给出客观、专业的投资建议。你的建议应当基于事实和数据，
同时考虑行业趋势、公司竞争力、财务健康状况等多方面因素。
请以JSON格式输出你的建议，包括投资评级、投资理由、风险因素、建议持有期限等关键信息。"""

    def _get_comparison_system_prompt(self) -> str:
        """获取比较分析系统提示词"""
        return """你是一位专业的财务分析师，擅长比较分析上市公司不同期间的财务报告，识别关键变化和趋势。
请根据提供的两期财报分析结果，进行全面的比较分析，重点关注财务指标的变化、业务发展趋势、盈利能力变化等。
你的分析应当客观、准确、专业，避免主观臈断，并以JSON格式输出结果。"""

    def _prepare_analysis_prompt(self, report_content: str, company_info: Dict[str, Any]) -> str:
        """准备财报分析提示词"""
        prompt = f"""请分析以下公司的财务报告，并提取关键财务指标和业务洞察：

公司信息：
- 公司名称：{company_info.get('company_name', '未知')}
- 股票代码：{company_info.get('stock_code', '未知')}
- 报告期间：{company_info.get('report_period', '未知')}

财报内容：
{report_content[:50000]}  # 限制长度以避免超出token限制

请提供以下分析结果（以JSON格式输出）：
1. 基本财务数据：总资产、总负债、净资产、营业收入、净利润、每股收益等
2. 盈利能力分析：毛利率、净利率、ROE、ROA等
3. 偿债能力分析：资产负债率、流动比率、速动比率等
4. 运营效率分析：应收账款周转率、存货周转率等
5. 现金流分析：经营活动、投资活动、筹资活动现金流
6. 业务亮点与风险：主要业务表现、风险因素等
7. 未来展望：公司战略、发展计划等
8. 分析摘要：对公司财务状况的总体评价（200-300字）

请确保输出为有效的JSON格式，键名使用英文，值可以使用中文。对于无法从报告中提取的指标，请标记为null。"""
        return prompt

    def _prepare_investment_prompt(self, analysis_result: Dict[str, Any]) -> str:
        """准备投资建议提示词"""
        # 将分析结果转换为字符串
        analysis_json = json.dumps(analysis_result, ensure_ascii=False, indent=2)
        
        prompt = f"""请基于以下财报分析结果，为投资者提供专业的投资建议：

财报分析结果：
{analysis_json}

请提供以下投资建议（以JSON格式输出）：
1. 投资评级：买入/增持/持有/减持/卖出
2. 目标价格：预期合理价格区间
3. 投资理由：支持你评级的关键理由（3-5点）
4. 风险因素：潜在的投资风险（3-5点）
5. 建议持有期限：短期/中期/长期
6. 适合投资者类型：保守型/稳健型/进取型
7. 投资建议摘要：总结性投资建议（200-300字）

请确保输出为有效的JSON格式，键名使用英文，值可以使用中文。"""
        return prompt

    def _prepare_comparison_prompt(self, current_report: Dict[str, Any], previous_report: Dict[str, Any]) -> str:
        """准备比较分析提示词"""
        # 将两期报告转换为字符串
        current_json = json.dumps(current_report, ensure_ascii=False, indent=2)
        previous_json = json.dumps(previous_report, ensure_ascii=False, indent=2)
        
        prompt = f"""请比较分析以下两期财报的结果，识别关键变化和趋势：

当期财报分析结果：
{current_json}

上期财报分析结果：
{previous_json}

请提供以下比较分析（以JSON格式输出）：
1. 财务指标变化：对比关键财务指标的变化（增长率、变动幅度等）
2. 盈利能力变化：毛利率、净利率、ROE等指标的变化趋势
3. 财务状况变化：资产负债结构、偿债能力的变化
4. 现金流变化：各类现金流的变化及原因分析
5. 业务发展趋势：主要业务的发展变化
6. 风险变化：新增或减轻的风险因素
7. 投资价值变化：投资价值的提升或降低
8. 比较分析摘要：对比分析的总体评价（200-300字）

请确保输出为有效的JSON格式，键名使用英文，值可以使用中文。"""
        return prompt

    def _parse_llm_response(self, response: Any) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 获取响应内容
            # litellm的响应格式与OpenAI类似，但可能有细微差别
            content = response.choices[0].message.content
            
            # 尝试解析JSON
            # 首先尝试直接解析
            try:
                result = json.loads(content)
                result["status"] = "success"
                return result
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试提取JSON部分
                import re
                json_pattern = r'```json\s*([\s\S]*?)\s*```'
                match = re.search(json_pattern, content)
                
                if match:
                    json_str = match.group(1)
                    try:
                        result = json.loads(json_str)
                        result["status"] = "success"
                        return result
                    except json.JSONDecodeError:
                        pass
                
                # 如果仍然失败，返回原始内容
                return {
                    "status": "partial",
                    "message": "无法解析为JSON格式",
                    "raw_content": content
                }
                
        except Exception as e:
            logger.error(f"解析LLM响应时出错: {str(e)}")
            return {
                "status": "error",
                "message": f"解析响应失败: {str(e)}"
            }

    def _parse_investment_response(self, response: Any) -> Dict[str, Any]:
        """解析投资建议响应"""
        # 使用与_parse_llm_response相同的逻辑
        return self._parse_llm_response(response)

    def _parse_comparison_response(self, response: Any) -> Dict[str, Any]:
        """解析比较分析响应"""
        # 使用与_parse_llm_response相同的逻辑
        return self._parse_llm_response(response)
        
    def _get_mock_investment_advice(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟的投资建议（用于演示）"""
        company_name = analysis_result.get("metadata", {}).get("company_name", "示例公司")
        stock_code = analysis_result.get("metadata", {}).get("stock_code", "000000")
        
        # 从分析结果中提取一些指标
        financial_indicators = analysis_result.get("financial_indicators", {})
        revenue_trend = financial_indicators.get("revenue", {}).get("trend", "neutral")
        profit_trend = financial_indicators.get("net_profit", {}).get("trend", "neutral")
        roe = financial_indicators.get("roe", {}).get("value", "15%")
        
        # 根据指标生成投资建议
        investment_rating = "增持" if revenue_trend == "up" and profit_trend == "up" else "持有"
        
        return {
            "status": "success",
            "investment_rating": investment_rating,
            "summary": f"基于对{company_name}（{stock_code}）财务状况的分析，我们给予"{investment_rating}"评级。公司展现出良好的盈利能力和成长性，净资产收益率达到{roe}，处于行业较高水平。",
            "investment_advice": {
                "short_term": "短期投资者可以在市场调整时逢低买入，目标价位上调至前期高点的1.1倍。",
                "medium_term": "中期投资者可以采取定投策略，分批建仓，持有6-12个月。",
                "long_term": "长期投资者可以将该股作为核心持仓，定期检视公司基本面变化。"
            },
            "risk_assessment": {
                "market_risk": "中等",
                "financial_risk": "较低",
                "operational_risk": "较低",
                "policy_risk": "中等"
            },
            "valuation": {
                "pe_ratio": "18.5",
                "pb_ratio": "2.3",
                "industry_comparison": "低于行业平均水平，具有一定的安全边际。",
                "fair_value_range": "当前股价的0.9-1.2倍"
            },
            "key_watch_points": [
                "关注公司新产品线的市场表现",
                "关注原材料价格波动对毛利率的影响",
                "关注行业政策变化",
                "关注海外市场拓展进度",
                "关注研发投入转化效率"
            ]
        }
    
    def _get_mock_comparison_result(self, current_report: Dict[str, Any], previous_report: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟的比较分析结果（用于演示）"""
        company_name = current_report.get("metadata", {}).get("company_name", "示例公司")
        current_period = current_report.get("metadata", {}).get("report_period", "2023年年度报告")
        previous_period = previous_report.get("metadata", {}).get("report_period", "2022年年度报告")
        
        return {
            "status": "success",
            "comparison_summary": f"{company_name}在{current_period}相比{previous_period}整体表现出积极的增长态势。营业收入和净利润均实现两位数增长，盈利能力稳中有升，财务结构保持稳健。",
            "financial_comparison": {
                "revenue": {
                    "current": "52.3亿元",
                    "previous": "45.4亿元",
                    "change": "+15.2%",
                    "trend": "up",
                    "analysis": "收入增长主要来自于新产品线的贡献和海外市场的拓展。"
                },
                "net_profit": {
                    "current": "8.7亿元",
                    "previous": "7.7亿元",
                    "change": "+12.8%",
                    "trend": "up",
                    "analysis": "利润增长略低于收入增长，主要受原材料成本上升影响。"
                },
                "gross_margin": {
                    "current": "32.5%",
                    "previous": "33.3%",
                    "change": "-0.8%",
                    "trend": "down",
                    "analysis": "毛利率小幅下降，主要受原材料价格上涨和市场竞争加剧影响。"
                },
                "roe": {
                    "current": "18.2%",
                    "previous": "17.7%",
                    "change": "+0.5%",
                    "trend": "up",
                    "analysis": "资产使用效率提升，带动净资产收益率小幅上升。"
                }
            },
            "business_comparison": {
                "market_share": {
                    "current": "18%",
                    "previous": "16.5%",
                    "change": "+1.5%",
                    "analysis": "市场份额稳步提升，品牌影响力增强。"
                },
                "product_mix": {
                    "current": "高端产品占比45%，中端产品占比40%，低端产品占比15%",
                    "previous": "高端产品占比40%，中端产品占比42%，低端产品占比18%",
                    "analysis": "产品结构持续优化，高端产品占比提升，有利于提高整体盈利能力。"
                },
                "r_and_d": {
                    "current": "研发投入4.2亿元，占收入8.0%",
                    "previous": "研发投入3.4亿元，占收入7.5%",
                    "analysis": "研发投入持续增加，研发强度提升，为未来增长奠定基础。"
                }
            },
            "trend_analysis": {
                "positive_trends": [
                    "收入规模持续扩大",
                    "盈利能力保持稳定",
                    "研发投入持续增加",
                    "高端产品占比提升",
                    "海外市场拓展加速"
                ],
                "negative_trends": [
                    "原材料成本上升压力增大",
                    "毛利率小幅下滑",
                    "市场竞争加剧",
                    "人力成本持续上升"
                ],
                "unchanged_factors": [
                    "主营业务方向保持稳定",
                    "财务结构保持稳健",
                    "股利政策保持连续"
                ]
            },
            "conclusion": f"{company_name}在{current_period}展现出良好的经营韧性和增长潜力。虽然面临成本上升和竞争加剧的挑战，但公司通过产品结构优化和市场拓展，实现了收入和利润的稳健增长。未来随着研发投入的持续加大和高端产品比例的提升，公司有望保持良好的增长态势。"
        }


# 使用示例
if __name__ == "__main__":
    # 设置API密钥（实际使用时应从环境变量或配置文件获取）
    api_key = "your-api-key"
    
    # 初始化服务
    llm_analyzer = LLMAnalysisService(api_key=api_key)
    
    # 示例财报内容和公司信息
    report_content = "这里是财报内容..."
    company_info = {
        "company_name": "示例公司",
        "stock_code": "000001",
        "report_period": "2023年年度报告"
    }
    
    # 分析财报
    analysis_result = llm_analyzer.analyze_financial_report(report_content, company_info)
    print(json.dumps(analysis_result, ensure_ascii=False, indent=2))
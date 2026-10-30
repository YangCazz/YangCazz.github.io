# 内容校验框架 —— 在构建期强制博客写作规范。
# 新文章(STRICT_FROM 起发布)违规 → error,中断构建;
# 存量文章(更早)违规 → warning,不阻断(用户批准"存量豁免新文强制")。
# 规范来源:CLAUDE.md「发布前检查清单」。
require 'date'
require 'set'
require 'nokogiri'
require 'kramdown'
require 'kramdown-parser-gfm'

module BlogContentValidator
  STRICT_FROM = Date.new(2026, 8, 2)
  module_function

  # 不允许“最后写入者获胜”：歧义别名会让校验与前端分类产生不同结果。
  def registry_index(registry)
    canonical = registry.map { |tag| tag['name'] }.to_set
    types = {}
    aliases = {}
    errors = []
    registry.each do |tag|
      name = tag['name']
      errors << "duplicate canonical tag '#{name}'" if types.key?(name)
      errors << "tag '#{name}' has invalid type '#{tag['type']}'" unless %w[domain method].include?(tag['type'])
      types[name] = tag['type']
      Array(tag['aliases']).each do |alias_name|
        if canonical.include?(alias_name) && alias_name != name
          errors << "alias '#{alias_name}' for '#{name}' conflicts with a canonical tag"
        elsif aliases.key?(alias_name) && aliases[alias_name] != name
          errors << "alias '#{alias_name}' maps to both '#{aliases[alias_name]}' and '#{name}'"
        else
          aliases[alias_name] = name
        end
      end
    end
    [canonical, types, aliases, errors]
  end

  # 示例中的 <cite> / ol.references 不是正文引用；支持四反引号包三反引号。
  def prose_without_fences(content)
    fence = nil
    content.each_line.map do |line|
      if fence
        fence = nil if line.match?(/\A {0,3}#{Regexp.escape(fence[0])}{#{fence[1]},}\s*\z/)
        "\n"
      elsif (match = line.match(/\A {0,3}(`{3,}|~{3,})/))
        fence = [match[1][0], match[1].length]
        "\n"
      else
        line
      end
    end.join.gsub(/<!--.*?-->/m, '')
  end

  def ordered_list_numbers(list)
    number = Integer(list['start'], exception: false) || 1
    list.element_children.select { |node| node.name == 'li' }.map do |item|
      number = Integer(item['value'], exception: false) || number
      current = number
      number += 1
      current
    end
  end

  def missing_citations(content)
    prose = prose_without_fences(content)
    document = Nokogiri::HTML.fragment(prose)
    document.css('pre, code').remove
    citations = document.css('cite').filter_map { |node| node.text.strip[/\A\[(\d+)\]\z/, 1]&.to_i }
    references = document.css('ol.references').flat_map { |list| ordered_list_numbers(list) }

    # Markdown 数字列表只在参考文献章节内计数，避免正文步骤列表掩盖缺失引用。
    section_level = nil
    reference_sections = +''
    prose.each_line do |line|
      if (heading = line.match(/\A {0,3}(\#{1,6})\s+(.+?)\s*#*\s*\z/))
        level = heading[1].length
        if heading[2].match?(/\A(?:参考文献|参考资料|参考链接|References|Bibliography)\z/i)
          section_level = level
        elsif section_level && level <= section_level
          section_level = nil
        end
      elsif section_level
        reference_sections << line
      end
    end
    rendered = Nokogiri::HTML.fragment(Kramdown::Document.new(reference_sections, input: 'GFM').to_html)
    rendered.css('ol').reject { |list| list.ancestors('ol').any? }.each do |list|
      references.concat(ordered_list_numbers(list))
    end
    (citations - references).uniq
  end
end

Jekyll::Hooks.register :site, :post_read do |site|
  registry = site.data['tags'] || []
  canonical, tag_types, alias_map, registry_errors = BlogContentValidator.registry_index(registry)
  errors = registry_errors.map { |error| "_data/tags.yml: #{error}" }
  warnings = []

  site.posts.docs.each do |post|
    legacy = (post.date.to_date < BlogContentValidator::STRICT_FROM)
    path = post.relative_path
    raw = post.content.to_s
    front = post.data.to_h
    fm = front.select { |k, _| %w[title date categories tags excerpt image].include?(k) }
    fm['categories'] = Array(front['categories'])
    fm['tags'] = Array(front['tags'])
    violations = []

    # R1: frontmatter 必填字段
    %w[title date categories excerpt].each do |f|
      v = fm[f]
      if v.nil? || (v.respond_to?(:empty?) && v.empty?)
        violations << "#{path}: missing frontmatter field '#{f}'"
      end
    end

    # R2: 日期格式 YYYY-MM-DD HH:MM:SS(.SSS) ±HHMM
    d = front['date'].to_s
    unless d =~ /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)? [+-]\d{2}:?\d{2}$/
      violations << "#{path}: date '#{d}' not matching YYYY-MM-DD HH:MM:SS.SSS +0800"
    end

    # R3: 标签校验(旧规则)
    tags = fm['tags']
    violations << "#{path}: has #{tags.length} tags (max 8)" if tags.length > 8
    resolved = tags.map do |tag|
      if canonical.include?(tag)
        tag
      elsif alias_map.key?(tag)
        warnings << "#{path}: tag '#{tag}' is an alias — use '#{alias_map[tag]}' instead"
        alias_map[tag]
      else
        violations << "#{path}: tag '#{tag}' is not registered in _data/tags.yml"
        nil
      end
    end.compact
    types = resolved.map { |t| tag_types[t] }.compact
    unless types.include?('domain')
      violations << "#{path}: missing a domain tag (problem area)" unless types.empty?
    end
    violations << "#{path}: missing a method tag (technique/tool)" unless types.include?('method')

    # R4: 标题禁止手动编号
    bad_heading = raw.scan(/^\#{2,6}\s+(?:[0-9]{1,2}[.、)])|^\#{2,6}\s+(?:[一二三四五六七八九十百]+[.、)])/).first
    if bad_heading
      violations << "#{path}: manually numbered heading '#{bad_heading.strip}' — use auto-numbering"
    end

    # R5: <cite>[N]</cite> 引用编号必须存在于参考文献列表
    missing = BlogContentValidator.missing_citations(raw)
    unless missing.empty?
      violations << "#{path}: cite references #{missing.uniq.join(',')} not found in reference list"
    end

    # R6: 图片路径必须在 /assets/images/ 下(排除外链)
    # 兼容 Liquid 包裹写法:![alt]({{ '/assets/images/...' | relative_url }}) —— 先解出路径再校验
    raw.scan(/!\[[^\]]*\]\(([^)]+)\)/).flatten.each do |url|
      u = url.strip
      if u.start_with?('{{')
        m = u.match(/\{\{['"]?([^'"|}]+)['"]?/)
        u = m ? m[1].strip : u
      end
      next if u.start_with?('/assets/images/', 'http://', 'https://')
      next if u.empty?
      violations << "#{path}: image path '#{u}' must be under /assets/images/"
    end

    # R7: 禁止 emoji
    if raw =~ /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}]/u
      violations << "#{path}: contains emoji characters (forbidden)"
    end

    # 存量豁免:存量文章违规 → warning;新文章违规 → error
    if legacy
      warnings.concat(violations.map { |v| v + ' [legacy]' })
    else
      errors.concat(violations)
    end
  end

  warnings.each { |w| Jekyll.logger.warn 'Content Validator:', w }

  if errors.any?
    Jekyll.logger.error 'Content Validator:', "Found #{errors.length} error(s):"
    errors.each { |e| Jekyll.logger.error '', "  - #{e}" }
    raise 'Content validation failed — fix the errors above and rebuild.'
  end

  Jekyll.logger.info 'Content Validator:', "All checks passed (#{site.posts.docs.length} posts)"
end

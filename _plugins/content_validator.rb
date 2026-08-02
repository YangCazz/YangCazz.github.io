# 内容校验框架 —— 在构建期强制博客写作规范。
# 新文章(STRICT_FROM 起发布)违规 → error,中断构建;
# 存量文章(更早)违规 → warning,不阻断(用户批准"存量豁免新文强制")。
# 规范来源:CLAUDE.md「发布前检查清单」。
require 'date'

Jekyll::Hooks.register :site, :post_read do |site|
  STRICT_FROM = Date.new(2026, 8, 2)  # 此日期起发布的文章严格校验;更早的存量文章宽松校验

  registry = site.data['tags'] || []
  canonical = registry.map { |t| t['name'] }.to_set
  tag_types = {}
  registry.each { |t| tag_types[t['name']] = t['type'] }
  alias_map = {}
  registry.each do |t|
    (t['aliases'] || []).each { |a| alias_map[a] = t['name'] }
  end

  errors = []
  warnings = []

  site.posts.docs.each do |post|
    legacy = (post.date.to_date < STRICT_FROM)
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
    cite_nums = raw.scan(/<cite>\[(\d+)\]<\/cite>/).flatten.map(&:to_i)
    ref_nums = raw.scan(/^(\d+)\.\s/).flatten.map(&:to_i)
    missing = cite_nums - ref_nums
    unless missing.empty?
      violations << "#{path}: cite references #{missing.uniq.join(',')} not found in reference list"
    end

    # R6: 图片路径必须在 /assets/images/ 下(排除外链)
    raw.scan(/!\[[^\]]*\]\(([^)]+)\)/).flatten.each do |url|
      next if url.start_with?('/assets/images/', 'http://', 'https://')
      next if url.strip.empty?
      violations << "#{path}: image path '#{url}' must be under /assets/images/"
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

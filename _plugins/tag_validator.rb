# Tag validation hook — runs after Jekyll reads all post frontmatter.
# Validates every post's tags against _data/tags.yml.
# Aborts build on: unknown tag, too many tags, missing domain or method tag.

Jekyll::Hooks.register :site, :post_read do |site|
  registry = site.data['tags'] || []
  if registry.empty?
    Jekyll.logger.warn "Tag Validator:", "_data/tags.yml is empty or missing — skipping validation"
    next
  end

  canonical_names = registry.map { |t| t['name'] }.to_set
  alias_map = {}
  tag_types = {}
  registry.each do |t|
    tag_types[t['name']] = t['type']
    (t['aliases'] || []).each { |a| alias_map[a] = t['name'] }
  end

  errors = []
  warnings = []

  site.posts.docs.each do |post|
    tags = post.data['tags'] || []
    post_path = post.relative_path

    # Check count cap
    if tags.length > 8
      errors << "#{post_path}: has #{tags.length} tags (max 8). Tags: #{tags.join(', ')}"
    end

    resolved = []
    tags.each do |tag|
      if canonical_names.include?(tag)
        resolved << tag
      elsif alias_map.key?(tag)
        resolved << alias_map[tag]
        warnings << "#{post_path}: tag '#{tag}' is an alias — use '#{alias_map[tag]}' instead"
      else
        errors << "#{post_path}: tag '#{tag}' is not registered in _data/tags.yml"
      end
    end

    # Check type balance
    types = resolved.map { |t| tag_types[t] }.compact
    has_domain = types.include?('domain')
    has_method = types.include?('method')
    if !has_domain && !has_method
      errors << "#{post_path}: has no registered tags after resolution"
    elsif !has_domain
      errors << "#{post_path}: missing a domain tag (problem area)"
    elsif !has_method
      errors << "#{post_path}: missing a method tag (technique/tool)"
    end
  end

  warnings.each { |w| Jekyll.logger.warn "Tag Validator:", w }

  if errors.any?
    Jekyll.logger.error "Tag Validator:", "Found #{errors.length} tag error(s):"
    errors.each { |e| Jekyll.logger.error "", "  - #{e}" }
    raise "Tag validation failed — fix the errors above and rebuild."
  end

  Jekyll.logger.info "Tag Validator:", "All tags valid (#{site.posts.docs.length} posts checked)"
end

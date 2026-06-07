require 'yaml'
require 'set'

# Load registry
registry = YAML.load_file('_data/tags.yml')
canonical = {}
registry.each do |t|
  canonical[t['name']] = t['type']
  (t['aliases'] || []).each { |a| canonical[a] = t['name'] }
end

Dir.glob('_posts/*.md').sort.each do |file|
  content = File.read(file)
  # Handle files with BOM
  content = content.sub("﻿", '')
  next unless content.start_with?('---')

  parts = content.split('---', 3)
  next if parts.length < 3

  fm = YAML.safe_load(parts[1], permitted_classes: [Time, Date])
  tags = fm['tags'] || []
  old_tags = tags.dup
  resolved = []

  tags.each do |tag|
    if canonical.key?(tag)
      canonical_name = canonical[tag]
      # If value is a type string, tag IS canonical name
      if %w[domain method].include?(canonical_name)
        resolved << tag unless resolved.include?(tag)
      else
        # Value is the canonical name (alias mapping)
        resolved << canonical_name unless resolved.include?(canonical_name)
      end
      # else: drop silently (unregistered single-use tag)
    end
  end

  fm['tags'] = resolved.uniq
  new_fm = YAML.dump(fm)
  # YAML.dump adds "---\n" at start, remove it
  new_fm = new_fm.sub(/\A---\n/, '')
  new_content = "---\n#{new_fm}---\n#{parts[2]}"
  File.write(file, new_content)

  puts "#{File.basename(file)}: #{old_tags.length}->#{resolved.length} tags  [#{resolved.join(', ')}]"
end
puts "\nMigration complete."
